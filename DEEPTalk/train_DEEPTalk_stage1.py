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
from utils.extra import seed_everything
from utils.loss import *
from utils.our_renderer import get_texture_from_template, render_flame, to_lip_reading_image, render_flame_lip, load_template_mesh, render_flame_nvdiff
from utils.nvdiff_util import *
import nvdiffrast.torch as dr
from torchvision import transforms,utils
import time
from tqdm import tqdm
## for implement DEE
sys.path.append(f'../')
from DEE.get_DEE import get_DEE_from_json
from FER.get_model import init_affectnet_feature_extractor
from DEE.utils.utils import compare_checkpoint_model

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
                    mask=None, affectnet_feature_extractor=None,log_wandb=True, tau=0.1):
    model.train()
    model.to(device)
    model.sequence_decoder.motion_prior.eval()
    train_loss = 0
    total_steps = len(data_loader)
    processor = data_loader.dataset.processor
    epoch_start = time.time()
    for i, data_label in enumerate(data_loader) :
        forward_start = time.time()
        data, label = data_label # [data (audio, flame_param), label]
        audio, flame_param = list_to(data, device) # (BS, T / 30 * 16000), (BS, T, 53) -> raw audio, flame parameters
        BS, T = flame_param.shape[:2]
        emotion, intensity, gender, actor_id = list_to(label, device)
        condition = label_to_condition_MEAD(config, emotion, intensity, actor_id).to(device)
        audio=audio.to(device)
        if model.__class__.__name__ == 'DEMOTE':
            params_pred = model(audio, condition)
        else:
            params_pred = model(audio, condition, tau=tau) # batch, seq_len, 53

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
            
            DEE_type = config["sequence_decoder_config"]["DEE"]["point_DEE"]
            num_samples = config["sequence_decoder_config"]["DEE"]["num_samples"]
            AV_emo_loss = calculate_consistency_loss(model.sequence_decoder.DEE, audio, exp_param_pred, 
                                                                  DEE_type, DEE_config.normalize_exp, num_samples,
                                                                  affectnet_feature_extractor=affectnet_feature_extractor)
            loss_dict['AV_emo_loss'] = AV_emo_loss.item()
            loss = loss + AV_emo_loss * config['loss']['AV_emo_loss']['weight']
            
        if config["loss"]["VV_emo_loss"]["use"] :
            DEE_type = config["sequence_decoder_config"]["DEE"]["point_DEE"]
            num_samples = config["sequence_decoder_config"]["DEE"]["num_samples"]
            VV_emo_loss = calculate_VV_emo_loss(model.sequence_decoder.DEE, exp_param_target, exp_param_pred, 
                                                                  DEE_type, DEE_config.normalize_exp, num_samples,
                                                                  affectnet_feature_extractor=affectnet_feature_extractor)
            loss_dict['VV_emo_loss'] = VV_emo_loss.item()
            loss = loss + VV_emo_loss * config['loss']['VV_emo_loss']['weight']
            
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
                    mask=None, affectnet_feature_extractor=None,log_wandb=True):
    model.to(device)
    model.eval()
    val_loss = 0
    total_steps = len(data_loader)
    processor = data_loader.dataset.processor
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
            if model.__class__.__name__ == 'DEMOTE':
                params_pred = model(audio, condition)
            else:
                params_pred = model(audio, condition,  tau=0.0001) # batch, seq_len, 53
            exp_param_pred = params_pred[:,:,:50].to(device)
            jaw_pose_pred = params_pred[:,:,50:53].to(device)
            exp_param_target = flame_param[:,:,:50].to(device)
            jaw_pose_target = flame_param[:,:,50:53].to(device)

            vertices_pred = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_pred, jaw_pose_pred, device)
            vertices_target = flame.get_vertices_from_flame(FLINT_config, FLAME, exp_param_target, jaw_pose_target, device)    
            recon_loss = calculate_vertice_loss(vertices_pred, vertices_target)
            val_loss_dict['recon_loss'] += recon_loss.item()
            loss = recon_loss.clone()
            
            if config["loss"]["vertex_velocity_loss"]["use"] :
                velocity_loss = calculate_vertex_velocity_loss(vertices_pred, vertices_target)
                val_loss_dict['velocity_loss'] += velocity_loss.item()
                loss = loss + velocity_loss
                
            if config["loss"]["vertex_lip_loss"]["use"] :
                B,T,V = vertices_target.shape
                vertices_target_ = vertices_target.reshape(B*T,-1,3)
                target_lip = mask.masked_vertice('lips', vertices_target_.shape, vertices_target_, device)
                vertices_pred_ = vertices_pred.reshape(B*T,-1,3)
                pred_lip = mask.masked_vertice('lips', vertices_pred_.shape, vertices_pred_, device)
                lip_loss = calculate_vertice_loss(pred_lip, target_lip)
                val_loss_dict['lip_loss'] += lip_loss.item()
                loss = loss + lip_loss
                
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
                
                DEE_type = config["sequence_decoder_config"]["DEE"]["point_DEE"]
                num_samples = config["sequence_decoder_config"]["DEE"]["num_samples"]
                AV_emo_loss = calculate_consistency_loss(model.sequence_decoder.DEE, audio, exp_param_pred, 
                                                                    DEE_type, DEE_config.normalize_exp, num_samples,
                                                                    affectnet_feature_extractor=affectnet_feature_extractor)
                val_loss_dict['AV_emo_loss'] += AV_emo_loss.item() # changed to raw
                loss = loss + AV_emo_loss * config['loss']['AV_emo_loss']['weight']
                
            if config["loss"]["VV_emo_loss"]["use"] :
                DEE_type = config["sequence_decoder_config"]["DEE"]["point_DEE"]
                num_samples = config["sequence_decoder_config"]["DEE"]["num_samples"]
                VV_emo_loss = calculate_VV_emo_loss(model.sequence_decoder.DEE, exp_param_target, exp_param_pred, 
                                                                    DEE_type, DEE_config.normalize_exp, num_samples,
                                                                    affectnet_feature_extractor=affectnet_feature_extractor)
                val_loss_dict['VV_emo_loss'] += VV_emo_loss.item()
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
    elif args.model_type == 'DEMOTE_vanila_VQ':
        TalkingHead = DEMOTE_VQ.DEMOTE_vanila_VQVAE(config, FLINT_config, DEE_config, FLINT_ckpt , DEE_model, load_motion_prior=config['motionprior_config']['load_motion_prior'])
    elif args.model_type == 'DEMOTE_VQ':
        TalkingHead = DEMOTE_VQ.DEMOTE_VQVAE(config, FLINT_config, DEE_config, FLINT_ckpt , DEE_model, load_motion_prior=config['motionprior_config']['load_motion_prior'])
    elif args.model_type == "DEMOTE_VQ_condition":
        TalkingHead = DEMOTE_VQ.DEMOTE_VQVAE_condition(config, FLINT_config, DEE_config, FLINT_ckpt , DEE_model, load_motion_prior=config['motionprior_config']['load_motion_prior'])
    TalkingHead = TalkingHead.to(device)
    # Load Mask for vertex loss
    mask = None
    if config["loss"]["vertex_lip_loss"]["use"] :
        mask = Mask(config['loss']['vertex_lip_loss']['mask_path'])

    if args.checkpoint is not None:
        print('loading checkpoint', args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        TalkingHead.load_state_dict(checkpoint)

    FLAME_train = flame.FLAME(config, batch_size=config["training"]["batch_size"]).to(device).eval()
    FLAME_val = flame.FLAME(config, batch_size=config["validation"]["batch_size"]).to(device).eval()
    FLAME_train.requires_grad_(False)
    FLAME_val.requires_grad_(False)

    print("Loading Dataset...")
    train_dataset = TalkingHeadDataset_new(config, split='train', process_audio=False)
    val_dataset = TalkingHeadDataset_new(config, split='val', process_audio=False)
    print('val_dataset', len(val_dataset),'| train_dataset', len(train_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["validation"]["batch_size"], drop_last=True)
    
    optimizer = torch.optim.Adam(TalkingHead.parameters(), lr=config["training"]["lr"])
    if config["training"]["scheduler"]:
        if config["training"]["scheduler"]["type"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler"]["step_size"], gamma=config["training"]["scheduler"]["gamma"])
    else:
        scheduler = None
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
                        tau=taus_per_epoch[epoch-1])
        print('training time for this epoch :', time.time() - training_time)

        save_path = os.path.join(config["training"]["save_dir"], config["name"], "EMOTE_{}.pth".format(epoch))
        if epoch % config["training"]["save_step"] == 0 :
            torch.save(TalkingHead.state_dict(), save_path)
            print("Save model at {}\n".format(save_path))

        validation_time = time.time()
        val_one_epoch(config, FLINT_config, DEE_config, epoch, TalkingHead, FLAME_val, val_dataloader, device,
                      mask=mask,
                      affectnet_feature_extractor=affectnet_feature_extractor, log_wandb=True)
        
        if scheduler is not None:
            scheduler.step()
        print('validation time for this epoch :', time.time() - validation_time)
        print("-"*50)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--EMOTE_config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, help = 'for stage2, we must give a checkpoint!')
    parser.add_argument('--DEE_checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['DEMOTE', 'DEMOTE_vanila_VQ', 'DEMOTE_VQ','DEMOTE_VQ_condition'])
    args = parser.parse_args()
    print(args)
    
    with open(args.EMOTE_config) as f:
        EMOTE_config = json.load(f)

    wandb.init(project = EMOTE_config["project_name"], # EMOTE
            name = EMOTE_config["name"], # test
            config = EMOTE_config) 
    
    save_dir = os.path.join(EMOTE_config["training"]["save_dir"], EMOTE_config["name"])
    os.makedirs(save_dir, exist_ok = True)
    
    save_arguments_to_file(save_dir)
    with open(f'{save_dir}/config.json', 'w') as f :
        json.dump(EMOTE_config, f)
    # # for debugging
    # EMOTE_config["data"]["smooth_expression"] = False
    main(args, EMOTE_config)

