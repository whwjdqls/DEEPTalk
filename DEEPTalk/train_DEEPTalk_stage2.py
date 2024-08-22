import argparse
import os
import glob
import torch
import json
import numpy as np
import wandb
import sys


os.environ["WANDB__SERVICE_WAIT"] = "300"
from datasets_.talkingheaddataset import TalkingHeadDataset_new
from utils.mask import Mask
from models import TVAE_inferno, DEMOTE, DEMOTE_VQ
from models.flame_models import flame
from utils.extra import seed_everything, Config
from utils.loss import *
from utils.our_renderer import get_texture_from_template, render_flame, to_lip_reading_image, render_flame_lip, load_template_mesh, render_flame_nvdiff
from utils.nvdiff_util import *
from utils.util import detect_landmarks, cut_mouth_vectorized
import nvdiffrast.torch as dr
from torchvision import transforms,utils
import time
from tqdm import tqdm
## for implement DEE
sys.path.append(f'../')
from DEE.get_DEE import get_DEE_from_json
from FER.get_model import init_affectnet_feature_extractor
from DEE.utils.utils import compare_checkpoint_model

## for Lipreading
import torchvision
# pip install face-alignment
import face_alignment
from models.auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer import E2E_new
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as t
from torchvision.utils import save_image
import yaml


def list_to(list_,device):
    """move a list of tensors to device
    """
    for i in range(len(list_)):
        list_[i] = list_[i].to(device)
    return list_

def label_to_condition_MEAD(config, emotion, intensity, actor_id):
    """labels to one hot condition vector
    """
    class_num_dict = config["sequence_decoder_config"]["style_embedding"]
    emotion = emotion - 1 # as labels start from 1
    emotion_one_hot = torch.nn.functional.one_hot(
        emotion, num_classes=class_num_dict["n_expression"]) #9
    intensity = intensity - 1 # as labels start from 1
    intensity_one_hot = torch.nn.functional.one_hot(
        intensity, num_classes=class_num_dict["n_intensities"]) #3
    # this might not be fair for validation set 
    actor_id_one_hot = torch.nn.functional.one_hot(
        actor_id, num_classes=class_num_dict["n_identities"]) # all actors #32
    condition = torch.cat([emotion_one_hot, intensity_one_hot, actor_id_one_hot], dim=-1) # (BS, 44)
    return condition.to(torch.float32)

def save_arguments_to_file(save_dir, filename='arguments.json'):
    save_path = save_dir + '/' + filename
    with open(save_path, 'w') as file:
        json.dump(vars(args), file)
    
def train_one_epoch(config,FLINT_config,DEE_config, epoch, model, FLAME, optimizer, data_loader, device, 
                    mask=None, affectnet_feature_extractor=None,log_wandb=True, new_lip_reading_model=None, anchor=None, tau=0.1):
    model.train()
    model.to(device)
    model.sequence_decoder.motion_prior.eval()
    train_loss = 0
    total_steps = len(data_loader)
    processor = data_loader.dataset.processor
    texture = None
    lip_reading_model = None
    epoch_start = time.time()
    is_point_DEE = True
    if model.sequence_decoder.DEE.__class__.__name__ == 'ProbDEE':
        is_point_DEE = False
    # initiate new lip reading model
    if config['loss']['lip_reading_loss']['face_detect'] :
        landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='sfd', device=str(device))

    for i, data_label in enumerate(data_loader) :
        forward_start = time.time()
        data, label = data_label # [data (audio, flame_param), label]
        audio, flame_param = list_to(data, device) # (BS, T / 30 * 16000), (BS, T, 53) -> raw audio, flame parameters
        BS, T = flame_param.shape[:2]
        emotion, intensity, gender, actor_id = list_to(label, device)
        condition = label_to_condition_MEAD(config, emotion, intensity, actor_id).to(device)
        audio=audio.to(device)
        params_pred = model(audio, condition, tau=tau) # batch, seq_len, 53
        
        if config["loss"]["swap"]["use"]: # swapping audio for swapped_vertices_pred
            batch_size, audio_size = audio.shape
            swapped_audio = audio.view(batch_size//2, 2, audio_size)
            # swap audio
            swapped_audio = swapped_audio[:,[1,0],:].view(batch_size, audio_size) # swap audio in batch 
            swapped_params_pred = model(audio, condition, audio_emotion=swapped_audio)

            swapped_exp_param_pred = swapped_params_pred[:,:,:50].to(device)
            swapped_jaw_pose_pred = swapped_params_pred[:,:,50:53].to(device)

            swapped_vertices_pred = flame.get_vertices_from_flame(
                FLINT_config, FLAME, swapped_exp_param_pred, swapped_jaw_pose_pred, device) # (BS, T, 15069)
            B,T,V = swapped_vertices_pred.shape
            swapped_vertices_pred = swapped_vertices_pred.reshape(B*T,-1,3)
        
        exp_param_pred = params_pred[:,:,:50].to(device)
        jaw_pose_pred = params_pred[:,:,50:53].to(device)
        exp_param_target = flame_param[:,:,:50].to(device)
        jaw_pose_target = flame_param[:,:,50:53].to(device)
        
        vertices_pred = flame.get_vertices_from_flame(
            FLINT_config, FLAME, exp_param_pred, jaw_pose_pred, device) # (BS, T, 15069)
        vertices_target = flame.get_vertices_from_flame(
            FLINT_config, FLAME, exp_param_target, jaw_pose_target, device) # (BS, T, 15069)
        loss_dict = {}
        recon_loss = calculate_vertice_loss(vertices_pred, vertices_target)
        loss = recon_loss.clone()
        loss_dict['recon_loss'] = recon_loss.item()
        
        if config["loss"]["swap"]["use"]: # second stage - swapping loss
            if DEE_config.process_type == 'layer_norm': #
                swapped_audio = torch.nn.functional.layer_norm(swapped_audio,(swapped_audio[0].shape[-1],))
            elif DEE_config.process_type == 'wav2vec2':
                swapped_audio = processor(swapped_audio,sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
            else:
                raise ValueError('DEE_config.process_type should be layer_norm or wav2vec2')
            if affectnet_feature_extractor is not None: # if DEE is using affectnet feature extractor
                swapped_exp_param_pred = torch.concat([swapped_exp_param_pred, swapped_jaw_pose_pred], dim=-1) # (Bs, T, 53)

            num_samples = config["sequence_decoder_config"]["DEE"]["num_samples"]
            swapped_emotion_consistency_loss = calculate_consistency_loss(model.sequence_decoder.DEE, swapped_audio, swapped_exp_param_pred,
                                                                          is_point_DEE,DEE_config.normalize_exp, num_samples,
                                                                          affectnet_feature_extractor=affectnet_feature_extractor)
            loss_dict['dis_emo'] = swapped_emotion_consistency_loss.item() # unweighted loss
            loss = loss + swapped_emotion_consistency_loss * config['loss']['swap']['weight']
            
   
            if texture is None: # load template mesh, texture, camera parameters once
                print('Initialize template mesh and camera parameters')
                glctx = dr.RasterizeCudaContext()
                print('Rasterizer loaded')
                uv, uv_idx, faces, texture = load_template_mesh('models/flame_models/geometry/head_template.obj', device)
                print('Initialize done')
                a_rot = np.array([[1, 0, 0, 0],
                                  [0,-1, 0, 0],
                                  [0, 0,-1, 0],
                                  [0, 0, 0, 1]]).astype(np.float32)
                a_translate = translate(0,0,0.2)
                proj =np.array([[1.7321,  0, 0, 0],
                                [0,  1.7321, 0, 0],
                                [0, 0, 1.0001, 1.0000],
                                [0,  0, -0.0100, 0]]).astype(np.float32)
                a_mv = np.matmul(a_translate, np.transpose(a_rot))
                a_mvp = np.matmul(np.transpose(proj), a_mv).astype(np.float32)
                a_mvp = torch.as_tensor(a_mvp, dtype=torch.float32, device=device)
                res = 224
                n_verts = 5023 # for flame -> mesh
                
            vertices_target = vertices_target.reshape(BS*T*n_verts, 3)
            vertices_target = torch.cat((vertices_target, torch.ones_like(vertices_target[..., 0:1])),dim=-1) # (BS*T,5023,4)
            vertices_target = torch.matmul(vertices_target, a_mvp.t()).reshape(BS*T,n_verts,4)
            vertices_pred = vertices_pred.reshape(BS*T*n_verts, 3)
            vertices_pred = torch.cat((vertices_pred, torch.ones_like(vertices_pred[..., 0:1])),dim=-1) # (BS*T, 5023, 3)
            vertices_pred = torch.matmul(vertices_pred, a_mvp.t()).reshape(BS*T,n_verts,4)
            swapped_vertices_pred = swapped_vertices_pred.reshape(BS*T*n_verts, 3)
            swapped_vertices_pred = torch.cat((swapped_vertices_pred, torch.ones_like(swapped_vertices_pred[..., 0:1])),dim=-1) #(BS*T, 5023, 3)
            swapped_vertices_pred = torch.matmul(swapped_vertices_pred, a_mvp.t()).reshape(BS*T,n_verts,4)
            start = time.time()
            images_target = render_flame_nvdiff(glctx, vertices_target, uv, uv_idx, faces, texture, res, device) # (BS*T,256, 256,4)
            images_pred = render_flame_nvdiff(glctx, vertices_pred, uv, uv_idx, faces, texture, res, device) # (BS*T,256, 256,4)
            swapped_images_pred = render_flame_nvdiff(glctx, swapped_vertices_pred, uv, uv_idx, faces, texture, res, device)
            
            images_target = images_target[...,:3].permute(0,3,1,2)# (BS*T,3,88, 88) 
            images_pred = images_pred[...,:3].permute(0,3,1,2)# (BS*T,3,88, 88)
            swapped_images_pred = swapped_images_pred[...,:3].permute(0,3,1,2)
            
            C,H,W = swapped_images_pred.shape[1:]
            images_target = images_target.view(BS,T,C,H,W) # (BS,T,3,224,224)
            images_pred = images_pred.view(BS,T,C,H,W) # (BS,T,3,224,224)
            swapped_images_pred = swapped_images_pred.view(BS,T,C,H,W)
            swapped_images_pred = swapped_images_pred.view(BS//2,2,T,C,H,W)
            swapped_images_pred = swapped_images_pred[:,[1,0],:,:,:,:].view(BS,T,C,H,W) # (BS,T,3,224,224)

            # print(f'image shape : {images_target.shape}')
            end = time.time()
            print(f'It takes {end-start}')

            if new_lip_reading_model is None:
                print(f'Using Original Lip reading Model')
                # load lip reading model
                if lip_reading_model is None :
                    lip_reading_model = LipReadingLoss(config['loss'], device, loss=config['loss']['lip_reading_loss']['metric'])
                    lip_reading_model.to(device).eval()
                    lip_reading_model.requires_grad_(False)

                # process lip images
                if config['loss']['lip_reading_loss']['face_detect'] :
                    # detect landmarks
                    landmarks_target = detect_landmarks(landmarks_detector, images_target, device)
                    landmarks_pred = detect_landmarks(landmarks_detector, images_pred, device)
                    landmarks_swap = detect_landmarks(landmarks_detector, swapped_images_pred, device)

                    # cut mouth images
                    lip_images_target = cut_mouth_vectorized(images_target, landmarks_target, device)
                    lip_images_pred = cut_mouth_vectorized(images_pred, landmarks_pred, device)
                    swapped_lip_images_pred = cut_mouth_vectorized(swapped_images_pred, landmarks_swap, device)
                else :
                    raise ValueError('Face detection should be True')
                    images_target = images_target[...,:3].permute(0,3,1,2)# (BS*T,3,88, 88) 
                    images_pred = images_pred[...,:3].permute(0,3,1,2)# (BS*T,3,88, 88)
                    swapped_images_pred = swapped_images_pred[...,:3].permute(0,3,1,2)
                    lip_images_target = to_lip_reading_image(images_target)#(BS*T, 1, 1, 88, 88)
                    lip_images_pred = to_lip_reading_image(images_pred)#(BS*T, 1, 1, 88, 88)
                    swapped_lip_images_pred = to_lip_reading_image(swapped_images_pred)
                    C,H,W = swapped_images_pred.shape[1:]

                # compute loss
                lip_loss = lip_reading_model(lip_images_target, lip_images_pred)
                disentangle_lip_loss = lip_reading_model(lip_images_target, swapped_lip_images_pred)
            
            else :
                if lip_reading_model is None :
                    print('New lip reading model loaded')
                    lip_reading_model = new_LipReadingLoss(device, new_lip_reading_model, convert_grayscale=True,
                                                            mouth_window_margin=12, mouth_crop_width=88,
                                                            mouth_crop_height=88, mouth_landmark_start_idx=48,
                                                            mouth_landmark_stop_idx=68)
                target_lip_embedding = lip_reading_model(images_target)
                pred_lip_embedding = lip_reading_model(images_pred)
                swapped_lip_embedding = lip_reading_model(swapped_images_pred)

                lip_loss = lip_reading_model.compute_feature_loss(target_lip_embedding,pred_lip_embedding)
                disentangle_lip_loss = lip_reading_model.compute_feature_loss(target_lip_embedding, swapped_lip_embedding)
                if config['loss']['new_lip_reading_loss']['anchor'] :
                    if config['loss']['new_lip_reading_loss']['compute_with_gt_and_masked_pred_lip'] :
                        compute_with_anchor = False
                    else : 
                        compute_with_anchor = True
                    anchor_lip_loss = lip_reading_model.compute_anchor_feature_loss(pred_lip_embedding, anchor, 0.6, target_lip_embedding, compute_with_anchor)
                    disentagle_anchor_lip_loss = lip_reading_model.compute_anchor_feature_loss(swapped_lip_embedding, anchor, 0.6, target_lip_embedding, compute_with_anchor)
                    loss_dict["anchor_lip_loss"] = anchor_lip_loss.item()
                    loss_dict["disentangle_anchor_lip_loss"] = disentagle_anchor_lip_loss.item()
                    loss += anchor_lip_loss * config['loss']['lip_reading_loss']['weight']
                    loss += disentagle_anchor_lip_loss * config['loss']['lip_reading_loss']['weight']

                elif config['loss']['new_lip_reading_loss']['jaw_filter'] :
                    jaw_filtered_lip_loss = lip_reading_model.compute_jaw_filter_loss(pred_lip_embedding, target_lip_embedding, jaw_pose_target, 0.15)
                    disentangle_jaw_filtered_lip_loss = lip_reading_model.compute_jaw_filter_loss(swapped_lip_embedding, target_lip_embedding, jaw_pose_target, 0.15)
                    loss_dict["jaw_filtered_lip_loss"] = jaw_filtered_lip_loss.item()
                    loss_dict["disentangle_jaw_filtered_lip_loss"] = disentangle_jaw_filtered_lip_loss.item()
                    loss += jaw_filtered_lip_loss * config['loss']['new_lip_reading_loss']['add_weight']
                    loss += disentangle_jaw_filtered_lip_loss * config['loss']['new_lip_reading_loss']['add_weight']

            loss_dict["lip_loss"] = lip_loss.item()
            loss = loss + lip_loss * config['loss']['lip_reading_loss']['weight']

            loss_dict['dis_lip_loss'] = disentangle_lip_loss.item()
            loss = loss + disentangle_lip_loss * config['loss']['lip_reading_loss']['weight']
            
        velocity_loss = torch.tensor(0.)
        if config["loss"]["vertex_velocity_loss"]["use"] :
            velocity_loss = calculate_vertex_velocity_loss(vertices_pred, vertices_target)
            loss = loss +  velocity_loss
            loss_dict['velocity_loss'] = velocity_loss.item()
            
        lip_loss = torch.tensor(0.)
        if config["loss"]["vertex_lip_loss"]["use"] :
            B,T,V = vertices_target.shape
            vertices_target_ = vertices_target.reshape(B*T,-1,3)
            target_lip = mask.masked_vertice('lips', vertices_target_.shape, vertices_target_, device)
            vertices_pred_ = vertices_pred.reshape(B*T,-1,3)
            pred_lip = mask.masked_vertice('lips', vertices_pred_.shape, vertices_pred_, device)
            lip_loss = calculate_vertice_loss(pred_lip, target_lip)
            loss = loss +  lip_loss
            loss_dict['lip_loss'] = lip_loss.item()
            
        # emotion consistency loss
        if affectnet_feature_extractor is not None: # if DEE is using affectnet feature extractor
            exp_param_pred = torch.concat([exp_param_pred, jaw_pose_pred], dim=-1) # (Bs, T, 53)
            exp_param_target = torch.concat([exp_param_target, jaw_pose_target], dim=-1) # (Bs, T, 53)

        if config["loss"]["AV_emo_loss"]["use"] :
            if DEE_config.process_type == 'layer_norm': #
                audio = torch.nn.functional.layer_norm(audio,(audio[0].shape[-1],))
            elif DEE_config.process_type == 'wav2vec2':
                audio = processor(audio,sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
            else:
                raise ValueError('DEE_config.process_type should be layer_norm or wav2vec2')
            
            is_point_DEE = config["sequence_decoder_config"]["DEE"]["point_DEE"]
            num_samples = config["sequence_decoder_config"]["DEE"]["num_samples"]
            AV_emo_loss = calculate_consistency_loss(model.sequence_decoder.DEE, audio, exp_param_pred, 
                                                                  is_point_DEE, DEE_config.normalize_exp, num_samples,
                                                                  affectnet_feature_extractor=affectnet_feature_extractor)
            loss_dict['AV_emo_loss'] = AV_emo_loss.item()
            loss = loss + AV_emo_loss * config['loss']['AV_emo_loss']['weight']
            
        if config["loss"]["VV_emo_loss"]["use"] :
            is_point_DEE = config["sequence_decoder_config"]["DEE"]["point_DEE"]
            num_samples = config["sequence_decoder_config"]["DEE"]["num_samples"]
            VV_emo_loss = calculate_VV_emo_loss(model.sequence_decoder.DEE, exp_param_target, exp_param_pred, 
                                                                  is_point_DEE, DEE_config.normalize_exp, num_samples,
                                                                  affectnet_feature_extractor=affectnet_feature_extractor)
            loss = loss + VV_emo_loss * config['loss']['VV_emo_loss']['weight']
            loss_dict['VV_emo_loss'] = VV_emo_loss.item()
            
            
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        loss_dict['loss'] = loss.item()
        train_loss += loss.detach().item()
        
        if i % config["training"]["log_step"] == 0:
            log_message = "Train Epoch: {} [{}/{} ({:.0f}%)]\t".format(epoch, i * BS, len(data_loader.dataset), 100.0 * i / len(data_loader))
            for key, value in loss_dict.items():
                log_message += "{}:{:.10f}, ".format(key, value)
            log_message += "time:{:.2f}".format(time.time()-forward_start)
            print(log_message)
        if log_wandb:
            wandb.log({f'train/{k} (step)':v for k,v in loss_dict.items()}) # for each step 
    if log_wandb:
        wandb.log({"train/loss (epoch)": train_loss / total_steps}) # for all 
    print("Train Epoch: {}\tAverage Loss: {:.10f}, time: {:.2f}".format(epoch, train_loss / total_steps, time.time() - epoch_start))

def val_one_epoch(config,FLINT_config,DEE_config, epoch, model, FLAME, data_loader, device, 
                    mask=None, affectnet_feature_extractor=None,log_wandb=True, new_lip_reading_model=None, anchor=None):
    model.to(device)
    model.eval()
    val_loss = 0
    total_steps = len(data_loader)
    texture=None
    lip_reading_model=None
    faces=None
    processor = data_loader.dataset.processor
    is_point_DEE = True
    # initiate new lip reading model
    if config['loss']['lip_reading_loss']['face_detect'] :
        landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='sfd', device=str(device))
    
    if model.sequence_decoder.DEE.__class__.__name__ == 'ProbDEE':
        is_point_DEE = False
    val_loss_dict = {'recon_loss':0, 'velocity_loss':0, 'lip_loss':0, 'AV_emo_loss':0, 'loss':0}
    with torch.no_grad():
        forward_start = time.time()
        for i, data_label in enumerate(tqdm(data_loader, desc="Processing", unit="step")) :
            data, label = data_label
            audio, flame_param = list_to(data, device)
            BS, T = flame_param.shape[:2]
            emotion, intensity, gender, actor_id = list_to(label, device)
            condition = label_to_condition_MEAD(config, emotion, intensity, actor_id).to(device)
            audio=audio.to(device)
            params_pred = model(audio, condition) 

            exp_param_pred = params_pred[:,:,:50].to(device)
            jaw_pose_pred = params_pred[:,:,50:53].to(device)
            exp_param_target = flame_param[:,:,:50].to(device)
            jaw_pose_target = flame_param[:,:,50:53].to(device)

            vertices_pred = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_pred, jaw_pose_pred, device)
            vertices_target = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_target, jaw_pose_target, device)    
            recon_loss = calculate_vertice_loss(vertices_pred, vertices_target)
            val_loss_dict['recon_loss'] += recon_loss.item()
            loss = recon_loss.clone()
            lip_loss = torch.tensor(0.)

            if texture is None: # load template mesh, texture, camera parameters once
                print('Initialize template mesh and camera parameters')
                # glctx = dr.RasterizeGLContext()
                glctx = dr.RasterizeCudaContext()
                uv, uv_idx, faces, texture = load_template_mesh('models/flame_models/geometry/head_template.obj', device)
                a_rot = np.array([[1, 0, 0, 0],
                                [0,-1, 0, 0],
                                [0, 0,-1, 0],
                                [0, 0, 0, 1]]).astype(np.float32)
                a_translate = translate(0,0,0.2)
                proj =np.array([[1.7321,  0, 0, 0],
                                [0,  1.7321, 0, 0],
                                [0, 0, 1.0001, 1.0000],
                                [0,  0, -0.0100, 0]]).astype(np.float32)
                # world -> cam
                a_mv = np.matmul(a_translate, np.transpose(a_rot))
                # 3D -> 2D
                a_mvp = np.matmul(np.transpose(proj), a_mv).astype(np.float32)
                a_mvp = torch.as_tensor(a_mvp, dtype=torch.float32, device=device)
                res = 224
                n_verts = 5023 # for flame -> mesh

            #  12-06 is reshaping okay?
            vertices_target = vertices_target.reshape(BS*T*n_verts, 3)
            vertices_target = torch.cat((vertices_target, torch.ones_like(vertices_target[..., 0:1])),dim=-1) # (BS*T,5023,4)
            vertices_target = torch.matmul(vertices_target, a_mvp.t()).reshape(BS*T,n_verts,4)
            vertices_pred = vertices_pred.reshape(BS*T*n_verts, 3)
            vertices_pred = torch.cat((vertices_pred, torch.ones_like(vertices_pred[..., 0:1])),dim=-1) # (BS*T, 5023, 3)
            vertices_pred = torch.matmul(vertices_pred, a_mvp.t()).reshape(BS*T,n_verts,4)
            
            images_target = render_flame_nvdiff(glctx, vertices_target, uv, uv_idx, faces, texture, res, device) # (BS*T,256, 256,4)
            images_pred = render_flame_nvdiff(glctx, vertices_pred, uv, uv_idx, faces, texture, res, device) # (BS*T,256, 256,4)
            images_target = images_target[...,:3].permute(0,3,1,2)# (BS*T,3,256, 256)
            images_pred = images_pred[...,:3].permute(0,3,1,2)# (BS*T,3,256, 256)

            # lip_images_target = to_lip_reading_image(images_target) #(BS*T, 1, 88, 88)
            # lip_images_pred = to_lip_reading_image(images_pred) #(BS*T, 1, 88, 88)
            # # for emotion loss
            C,H,W = images_target.shape[1:]
            images_target = images_target.view(BS,T,C,H,W) # (BS,T,3,224,224)
            images_pred = images_pred.view(BS,T,C,H,W) # (BS,T,3,224,224)

            if new_lip_reading_model is None:
                print(f'Using Original Lip reading Model')
                # load lip reading model
                if lip_reading_model is None :
                    lip_reading_model = LipReadingLoss(config['loss'], device, loss=config['loss']['lip_reading_loss']['metric'])
                    lip_reading_model.to(device).eval()
                    lip_reading_model.requires_grad_(False)
                # process lip images
                if config['loss']['lip_reading_loss']['face_detect'] :
                    # detect landmarks
                    landmarks_target = detect_landmarks(landmarks_detector, images_target, device)
                    landmarks_pred = detect_landmarks(landmarks_detector, images_pred, device)
                    # cut mouth images
                    lip_images_target = cut_mouth_vectorized(images_target, landmarks_target, device)
                    lip_images_pred = cut_mouth_vectorized(images_pred, landmarks_pred, device)
                else :
                    images_target = images_target[...,:3].permute(0,3,1,2)# (BS*T,3,88, 88) 
                    images_pred = images_pred[...,:3].permute(0,3,1,2)# (BS*T,3,88, 88)
                    lip_images_target = to_lip_reading_image(images_target)#(BS*T, 1, 1, 88, 88)
                    lip_images_pred = to_lip_reading_image(images_pred)#(BS*T, 1, 1, 88, 88)
                    C,H,W = images_pred.shape[1:]
                # compute loss
                lip_loss = lip_reading_model(lip_images_target, lip_images_pred)

            else :
                if lip_reading_model is None :
                    lip_reading_model = new_LipReadingLoss(device, new_lip_reading_model, convert_grayscale=True,
                                                            mouth_window_margin=12, mouth_crop_width=88,
                                                            mouth_crop_height=88, mouth_landmark_start_idx=48,
                                                            mouth_landmark_stop_idx=68)
                target_lip_embedding = lip_reading_model(images_target)
                pred_lip_embedding = lip_reading_model(images_pred)

                lip_loss = lip_reading_model.compute_feature_loss(target_lip_embedding,pred_lip_embedding)
                if config['loss']['new_lip_reading_loss']['anchor'] :
                    if config['loss']['new_lip_reading_loss']['compute_with_gt_and_masked_pred_lip'] :
                        compute_with_anchor = False
                    else : 
                        compute_with_anchor = True
                    anchor_lip_loss = lip_reading_model.compute_anchor_feature_loss(pred_lip_embedding, anchor, 0.6, target_lip_embedding, compute_with_anchor)
                    val_loss_dict['anchor_lip_loss'] = anchor_lip_loss
                    loss += anchor_lip_loss * config['loss']['lip_reading_loss']['weight']
                elif config['loss']['new_lip_reading_loss']['jaw_filter'] :
                    jaw_filtered_lip_loss = lip_reading_model.compute_jaw_filter_loss(pred_lip_embedding, target_lip_embedding, jaw_pose_target, 0.15)
                    val_loss_dict['jaw_filtered_lip_loss'] = jaw_filtered_lip_loss
                    loss += jaw_filtered_lip_loss*config['loss']['new_lip_reading_loss']['add_weight']

            val_loss_dict['lip_loss'] += lip_loss.item()
            loss += lip_loss * config['loss']['lip_reading_loss']['weight']
            
            if config["loss"]["vertex_velocity_loss"]["use"] :
                velocity_loss = calculate_vertex_velocity_loss(vertices_pred, vertices_target)
                val_loss_dict['velocity_loss'] = velocity_loss.item()
                loss = loss + velocity_loss
                
            if config["loss"]["vertex_lip_loss"]["use"] :
                B,T,V = vertices_target.shape
                vertices_target_ = vertices_target.reshape(B*T,-1,3)
                target_lip = mask.masked_vertice('lips', vertices_target_.shape, vertices_target_, device)
                vertices_pred_ = vertices_pred.reshape(B*T,-1,3)
                pred_lip = mask.masked_vertice('lips', vertices_pred_.shape, vertices_pred_, device)
                lip_loss = calculate_vertice_loss(pred_lip, target_lip)
                val_loss_dict['lip_loss'] = lip_loss.item()
                loss = loss + lip_loss
                
        # emotion consistency loss
            if affectnet_feature_extractor is not None: # if DEE is using affectnet feature extractor
                exp_param_pred = torch.concat([exp_param_pred, jaw_pose_pred], dim=-1) # (Bs, T, 53)
                exp_param_target = torch.concat([exp_param_target, jaw_pose_target], dim=-1) # (Bs, T, 53)


            if DEE_config.process_type == 'layer_norm': #
                audio = torch.nn.functional.layer_norm(audio,(audio[0].shape[-1],))
            elif DEE_config.process_type == 'wav2vec2':
                audio = processor(audio,sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
            else:
                raise ValueError('DEE_config.process_type should be layer_norm or wav2vec2')
            
            num_samples = config["sequence_decoder_config"]["DEE"]["num_samples"]
            AV_emo_loss = calculate_consistency_loss(model.sequence_decoder.DEE, audio, exp_param_pred, 
                                                                is_point_DEE, DEE_config.normalize_exp, num_samples,
                                                                affectnet_feature_extractor=affectnet_feature_extractor)
            val_loss_dict['AV_emo_loss'] += AV_emo_loss.item() # unweighted loss
            
            if config["loss"]["AV_emo_loss"]["use"]:
                loss = loss + AV_emo_loss * config['loss']['AV_emo_loss']['weight']
                
            VV_emo_loss = calculate_VV_emo_loss(model.sequence_decoder.DEE, exp_param_target, exp_param_pred, 
                                                                is_point_DEE, DEE_config.normalize_exp, num_samples,
                                                                affectnet_feature_extractor=affectnet_feature_extractor)
            val_loss_dict['VV_emo_loss'] = VV_emo_loss.item()# unweighted loss
            
            if config["loss"]["VV_emo_loss"]["use"] :
                loss = loss + VV_emo_loss * config['loss']['VV_emo_loss']['weight']

            val_loss_dict['loss'] += loss.item()
    if log_wandb:
        wandb.log({f'val/{k} (epoch)':v/total_steps for k,v in val_loss_dict.items()}) # for each epoch
    print("Val Epoch: {}\tAverage Loss: {:.10f}, time: {:.2f}".format(epoch, val_loss_dict['loss']/total_steps, time.time() - forward_start))

    
def main(args, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device', device)
    seed_everything(42)
    # loading FLINT checkpoint 
    FLINT_config_path = config['motionprior_config']['config_path']
    with open(FLINT_config_path) as f :
        FLINT_config = json.load(f) 
    FLINT_ckpt = config['motionprior_config']['checkpoint_path'] # FLINT trained model checkpoint path

    print("Loading Models...")
    # load DEE
    DEE_config_path = glob.glob(f'{os.path.dirname(args.DEE_checkpoint)}/*.json')[0]
    print(f'DEE config loaded :{DEE_config_path}') # DEE_config.use_affect_net
    DEE_model,DEE_config = get_DEE_from_json(DEE_config_path)
    DEE_checkpoint = torch.load(args.DEE_checkpoint, map_location='cpu')
    DEE_model.load_state_dict(DEE_checkpoint)
    DEE_model.eval()
    DEE_model.requires_grad_(False)
    compare_checkpoint_model(DEE_checkpoint, DEE_model.to('cpu'))
    DEE_model.to(device)
    
    # load affectnet feature extractor
    affectnet_feature_extractor = None
    if DEE_config.affectnet_model_path:
        model_path = DEE_config.affectnet_model_path
        config_path = os.path.dirname(model_path) + '/config.yaml'
        _, affectnet_feature_extractor = init_affectnet_feature_extractor(config_path, model_path)
        affectnet_feature_extractor.to(device)
        affectnet_feature_extractor.eval()
        affectnet_feature_extractor.requires_grad_(False)
    
    # load talkinghead model
    if args.model_type == 'DEMOTE':
        TalkingHead = DEMOTE.DEMOTE(config, FLINT_config, DEE_config, FLINT_ckpt , DEE_model, load_motion_prior=config['motionprior_config']['load_motion_prior'])
    elif args.model_type == 'DEMOTE_VQ':
        TalkingHead = DEMOTE_VQ.DEMOTE_VQVAE(config, FLINT_config, DEE_config, FLINT_ckpt , DEE_model, load_motion_prior=config['motionprior_config']['load_motion_prior'])
    elif args.model_type == "DEMOTE_VQ_condition":
        TalkingHead = DEMOTE_VQ.DEMOTE_VQVAE_condition(config, FLINT_config, DEE_config, FLINT_ckpt , DEE_model, load_motion_prior=config['motionprior_config']['load_motion_prior'])
    TalkingHead = TalkingHead.to(device)
    
    for params in TalkingHead.audio_model.parameters():
        params.requires_grad = False
    for params in TalkingHead.sequence_decoder.DEE.parameters():
        params.requires_grad = False

    # Load Mask for vertex loss
    mask = None
    if config["loss"]["vertex_lip_loss"]["use"] :
        mask = Mask(config['loss']['vertex_lip_loss']['mask_path'])

    if args.checkpoint is not None:
        print('loading checkpoint', args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        TalkingHead.load_state_dict(checkpoint)
    
    # Load FLAME
    FLAME_train = flame.FLAME(config, batch_size=config["training"]["batch_size"]).to(device).eval()
    FLAME_val = flame.FLAME(config, batch_size=config["validation"]["batch_size"]).to(device).eval()
    FLAME_train.requires_grad_(False)
    FLAME_val.requires_grad_(False)

    # Load New Lip reading model
    anchor=None
    if config["loss"]["new_lip_reading_loss"]["use"] :
        with open('models/auto_avsr/configs/model/visual_backbone/resnet_conformer.yaml', 'r') as file:
            config = yaml.safe_load(file)
        config = Config(**config)
        lip_reading_model = E2E_new(5049, config)
        lip_reading_ckpt = torch.load('models/auto_avsr/pretrained_model/vsr_trlrwlrs2lrs3vox2avsp_base.pth')
        lip_reading_model.load_state_dict(lip_reading_ckpt)
        lip_reading_model.to(device).eval()
    else :
        lip_reading_model = None


    # Load dataset
    print("Loading Dataset...")
    train_dataset = TalkingHeadDataset_new(config, split='train', process_audio=False)
    val_dataset = TalkingHeadDataset_new(config, split='val', process_audio=False)
    print('val_dataset', len(val_dataset),'| train_dataset', len(train_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["validation"]["batch_size"], drop_last=True)
    
    optimizer = torch.optim.Adam(TalkingHead.parameters(), lr=config["training"]["lr"])
    save_dir = os.path.join(config["training"]["save_dir"], config["name"])
    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    taus_per_epoch = np.linspace(config["training"]["tau_schedule"]["start"], config["training"]["tau_schedule"]["end"], config["training"]["num_epochs"])
    
    for epoch in range(1, config["training"]['num_epochs']+1):
        print('epoch', epoch, 'num_epochs', config["training"]['num_epochs'])

        training_time = time.time()
        train_one_epoch(config, FLINT_config, DEE_config, epoch, TalkingHead, FLAME_train, optimizer, train_dataloader, device,
                        mask=mask, 
                        affectnet_feature_extractor=affectnet_feature_extractor,log_wandb=True, 
                        new_lip_reading_model=lip_reading_model,
                        tau=taus_per_epoch[epoch-1])
        print('training time for this epoch :', time.time() - training_time)

        save_path = os.path.join(config["training"]["save_dir"], config["name"], "EMOTE_{}.pth".format(epoch))
        if epoch % config["training"]["save_step"] == 0 :
            torch.save(TalkingHead.state_dict(), save_path)
            print("Save model at {}\n".format(save_path))

        validation_time = time.time()
        val_one_epoch(config, FLINT_config, DEE_config, epoch, TalkingHead, FLAME_val, val_dataloader, device,
                      mask=mask,
                      affectnet_feature_extractor=affectnet_feature_extractor, log_wandb=True, 
                      new_lip_reading_model=lip_reading_model)
        print('validation time for this epoch :', time.time() - validation_time)
        print("-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--EMOTE_config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, help = 'for stage2, we must give a checkpoint!')
    parser.add_argument('--DEE_checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['DEMOTE', 'DEMOTE_VQ','DEMOTE_VQ_condition'])
    args = parser.parse_args()
    print(args)
    
    with open(args.EMOTE_config) as f:
        EMOTE_config = json.load(f)

    wandb.init(project = EMOTE_config["project_name"], # EMOTE
            name = EMOTE_config["name"], # test
            config = EMOTE_config) 
    
    save_dir = os.path.join(EMOTE_config["training"]["save_dir"], EMOTE_config["name"])
    # save_dir = os.path.join('/home/jisoo6687/EMOTE/checkpoints/DEMOTE', EMOTE_config["name"])
    os.makedirs(save_dir, exist_ok = True)
    
    save_arguments_to_file(save_dir)
    
    with open(f'{save_dir}/config.json', 'w') as f :
        json.dump(EMOTE_config, f)

    main(args, EMOTE_config)

