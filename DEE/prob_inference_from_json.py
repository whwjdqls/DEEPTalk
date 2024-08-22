import os
import torch

import argparse
from config import get_args_parser
import models
import datasets
from utils.loss import ClosedFormSampledDistanceLoss
from utils.prob_eval import compute_csd_sims,compute_matching_prob_sims
from utils.pcme import sample_gaussian_tensors
from torch.utils.data import DataLoader
import tqdm
import time
import json
import random
import glob
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
# import seaborn as sns
import matplotlib.pyplot as plt
import evaluation
import visualize
import utils 
from utils.utils import compare_checkpoint_model
from transformers import Wav2Vec2Processor
from datetime import datetime
torch.cuda.empty_cache()
import sys
sys.path.append('../')
from FER.get_model import init_affectnet_feature_extractor
def json2args(json_path):
    with open(json_path) as f:
        args = argparse.Namespace(**json.load(f))
    return args

@torch.no_grad()
def inference_one_epoch(args, model, val_dataloader , device, processor):
    cumulative_loss = 0
    model.eval()
    criterion = model.criterion
    exp_means = []
    audio_means = []
    exp_sigmas = []
    audio_sigmas = []
    emotion_list = []
    intensity_list = []
    gender_list = []
    actor_list = []
    if args.affectnet_model_path is not None:        
        model_path = args.affectnet_model_path
        config_path = os.path.dirname(model_path) + '/config.yaml'
        cfg, affectnet_feature_extractor = init_affectnet_feature_extractor(config_path,model_path)
        assert cfg.model.layers[-2] == args.parameter_dim, f"cfg.model.layers[-2] : {cfg.model.layers[-2]} != args.parameter_dim : {args.parameter_dim}"
        affectnet_feature_extractor.to(device)
        affectnet_feature_extractor.eval()
        affectnet_feature_extractor.requires_grad_(False)
    for samples, labels in tqdm.tqdm(val_dataloader):
        '''
        samples : [audio_processed,expression_processed]
        labels : [emotion, intensity, gender, actor_id] ->[ int, int, int, int]
        '''
        if args.process_type == 'wav2vec2':
            audio_samples = processor(samples[0],sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
        elif args.process_type == 'layer_norm':
            audio_samples = torch.nn.functional.layer_norm(samples[0],(samples[0].shape[-1],)).to(device)
        else:
            raise ValueError("process_type not supported")
        expression_samples = samples[1].to(device)
        if args.affectnet_model_path:
            expression_samples = affectnet_feature_extractor.extract_feature_from_layer(expression_samples, layer_num = -2)
            if args.normalize_affectnet_features: # normalize on the temporal axis
                expression_samples = (expression_samples - expression_samples.mean(dim=1, keepdim=True)) / expression_samples.std(dim=1, keepdim=True)
        audio_embedding, expression_embedding = model(audio_samples, expression_samples)

        exp_means.append(expression_embedding['mean'].detach().cpu().numpy())
        exp_sigmas.append(expression_embedding['std'].detach().cpu().numpy())
        audio_means.append(audio_embedding['mean'].detach().cpu().numpy())
        audio_sigmas.append(audio_embedding['std'].detach().cpu().numpy())
        emotion, intensity, gender, actor_name = labels
        
        emotion = torch.tensor(emotion).unsqueeze(1) #(BS,1)
        intensity = torch.tensor(intensity).unsqueeze(1) #(BS,1)
        gender = torch.tensor(gender).unsqueeze(1)
        actor_name = torch.tensor(actor_name).unsqueeze(1)
        
        
        emotion_list.append(emotion)
        intensity_list.append(intensity)
        gender_list.append(gender)
        actor_list.append(actor_name) 

        batch_size = len(audio_samples)
        matched = torch.eye(batch_size).to(device)
        loss, loss_dict = criterion(expression_embedding, audio_embedding, matched)
        cumulative_loss += loss.item() #

    print(f'avg loss:', cumulative_loss / len(val_dataloader))
    audio_means = np.concatenate(audio_means, axis=0)
    audio_sigmas = np.concatenate(audio_sigmas, axis=0)
    exp_means = np.concatenate(exp_means, axis=0)
    exp_sigmas = np.concatenate(exp_sigmas, axis=0)
    
    emotion = torch.cat(emotion_list, dim=0) # (validation_size,1)
    intensity = torch.cat(intensity_list, dim=0) # (validation_size,1)
    gender = torch.cat(gender_list, dim = 0) # (validation_size,1)
    actor = torch.cat(actor_list, dim = 0) # (validation_size,1)
    
    DB_emositygen = torch.cat((emotion, intensity, gender), dim=1) # (val_size,3)
    print("DB_emositygen shape: ", DB_emositygen.shape)
    print("actor shape: ", actor.shape)
    print("emotion shape: ", emotion.shape)
    print("intensity shape: ", intensity.shape)
    print("gender shape", gender.shape)
    
    DB_labels = torch.cat((emotion, intensity, gender, actor), dim=1).detach().cpu().numpy()
    print(f'Computing sims...')
    now = datetime.now()
    if args.inference_method == 'sampling':
        sampled_exp_features = sample_gaussian_tensors(torch.tensor(exp_means), torch.tensor(exp_sigmas), args.num_samples)
        sampled_audio_features = sample_gaussian_tensors(torch.tensor(audio_means), torch.tensor(audio_sigmas), args.num_samples)
        sims = compute_matching_prob_sims(
            sampled_exp_features, sampled_audio_features, 8,
            criterion.negative_scale, criterion.shift)
        
    elif args.inference_method == 'csd':
        sims = compute_csd_sims(exp_means, audio_means, exp_sigmas, audio_sigmas).T
    print(sims)
    print(f'Computing sims {sims.shape=} takes {datetime.now() - now}')  

    exp_retrieve_accuracy, audio_retrieve_accuracy, matched_audio, matched_expression = evaluation.cross_retrival_accuracy(sims, DB_emositygen, visualize=True)
    visualize.visualize_retrieval(exp_retrieve_accuracy, audio_retrieve_accuracy, matched_audio, matched_expression, args)
    visualize.visualize_embeddings(args,  exp_means, audio_means, DB_emositygen, method = 'both')
    visualize.visualize_gap(args, exp_means, audio_means, DB_emositygen, method = 'both')
    return exp_means, audio_means, exp_sigmas, audio_sigmas, DB_emositygen, sims,DB_labels

@torch.no_grad()
def inference_one_epoch_wo_visualization(args, model, val_dataloader , device, processor):
    cumulative_loss = 0
    model.eval()
    criterion = model.criterion
    exp_means = []
    audio_means = []
    exp_sigmas = []
    audio_sigmas = []
    emotion_list = []
    intensity_list = []
    gender_list = []
    actor_list = []
    if args.affectnet_model_path is not None:        
        model_path = args.affectnet_model_path
        config_path = os.path.dirname(model_path) + '/config.yaml'
        cfg, affectnet_feature_extractor = init_affectnet_feature_extractor(config_path,model_path)
        assert cfg.model.layers[-2] == args.parameter_dim, f"cfg.model.layers[-2] : {cfg.model.layers[-2]} != args.parameter_dim : {args.parameter_dim}"
        affectnet_feature_extractor.to(device)
        affectnet_feature_extractor.eval()
        affectnet_feature_extractor.requires_grad_(False)
    for samples, labels in tqdm.tqdm(val_dataloader):
        '''
        samples : [audio_processed,expression_processed]
        labels : [emotion, intensity, gender, actor_id] ->[ int, int, int, int]
        '''
        if args.process_type == 'wav2vec2':
            audio_samples = processor(samples[0],sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
        elif args.process_type == 'layer_norm':
            audio_samples = torch.nn.functional.layer_norm(samples[0],(samples[0].shape[-1],)).to(device)
        else:
            raise ValueError("process_type not supported")
        expression_samples = samples[1].to(device)
        if args.affectnet_model_path:
            expression_samples = affectnet_feature_extractor.extract_feature_from_layer(expression_samples, layer_num = -2)
            if args.normalize_affectnet_features: # normalize on the temporal axis
                expression_samples = (expression_samples - expression_samples.mean(dim=1, keepdim=True)) / expression_samples.std(dim=1, keepdim=True)
        audio_embedding, expression_embedding = model(audio_samples, expression_samples)

        exp_means.append(expression_embedding['mean'].detach().cpu().numpy())
        exp_sigmas.append(expression_embedding['std'].detach().cpu().numpy())
        audio_means.append(audio_embedding['mean'].detach().cpu().numpy())
        audio_sigmas.append(audio_embedding['std'].detach().cpu().numpy())
        emotion, intensity, gender, actor_name = labels
        
        emotion = torch.tensor(emotion).unsqueeze(1) #(BS,1)
        intensity = torch.tensor(intensity).unsqueeze(1) #(BS,1)
        gender = torch.tensor(gender).unsqueeze(1)
        actor_name = torch.tensor(actor_name).unsqueeze(1)
        
        
        emotion_list.append(emotion)
        intensity_list.append(intensity)
        gender_list.append(gender)
        actor_list.append(actor_name) 

        batch_size = len(audio_samples)
        matched = torch.eye(batch_size).to(device)
        loss, loss_dict = criterion(expression_embedding, audio_embedding, matched)
        cumulative_loss += loss.item() #

    print(f'avg loss:', cumulative_loss / len(val_dataloader))
    audio_means = np.concatenate(audio_means, axis=0)
    audio_sigmas = np.concatenate(audio_sigmas, axis=0)
    exp_means = np.concatenate(exp_means, axis=0)
    exp_sigmas = np.concatenate(exp_sigmas, axis=0)
    
    emotion = torch.cat(emotion_list, dim=0) # (validation_size,1)
    intensity = torch.cat(intensity_list, dim=0) # (validation_size,1)
    gender = torch.cat(gender_list, dim = 0) # (validation_size,1)
    actor = torch.cat(actor_list, dim = 0) # (validation_size,1)
    
    DB_emositygen = torch.cat((emotion, intensity, gender), dim=1) # (val_size,3)
    print("DB_emositygen shape: ", DB_emositygen.shape)
    print("actor shape: ", actor.shape)
    print("emotion shape: ", emotion.shape)
    print("intensity shape: ", intensity.shape)
    print("gender shape", gender.shape)
    
    DB_labels = torch.cat((emotion, intensity, gender, actor), dim=1).detach().cpu().numpy()
    sims = None
    return exp_means, audio_means, exp_sigmas, audio_sigmas, DB_emositygen, sims,DB_labels
def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training start using {device}...")
    
    #Choose number of epoch for checkpoint
    if args.last_ckpt :
        checkpoints = glob.glob(f'{args.save_dir}/*.pt')
        checkpoints = [os.path.basename(checkpoint) for checkpoint in checkpoints if "best.pt" not in checkpoint]
        if len(checkpoints[0].split('_')) == 2: # checkpoints like model_1.pt
            sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            epoch = sorted_checkpoints[-1].split('_')[-1].split('.')[0]
        elif len(checkpoints[0].split('_')) == 3: # checkpoints like model_1_0.011.pt
            sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-2]))
            epoch = sorted_checkpoints[-1].split('_')[-2]
    elif args.best_ckpt :
        epoch = 'best'
    else :
        epoch = args.num_ckpt
        
    print("Loading models...")
    DEE = models.ProbDEE(args)
    DEE = DEE.to(device)
    if args.affectnet_model_path:
        model_path = args.affectnet_model_path
        config_path = os.path.dirname(model_path) + '/config.yaml'
        cfg, affectnet_feature_extractor = init_affectnet_feature_extractor(config_path,model_path)
        assert cfg.model.layers[-2] == args.parameter_dim, f"cfg.model.layers[-2] : {cfg.model.layers[-2]} != args.parameter_dim : {args.parameter_dim}"
        affectnet_feature_extractor.to(device)
        affectnet_feature_extractor.eval()
        affectnet_feature_extractor.requires_grad_(False)
    print("Validation starts...")
    # Load checkpoint file
    checkpoint_path = glob.glob(f'{args.save_dir}/model_{epoch}*.pt')[0]

    print(f"Load checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path) 
    DEE.load_state_dict(checkpoint)

    if compare_checkpoint_model(checkpoint, DEE):
        print("Checkpoint and model are the same")
    else:
        print("Checkpoint and model are not the same")
        raise ValueError
        
        
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    start_time = time.time()
    dataset = datasets.AudioExpressionDataset(args, dataset=args.dataset,split = args.split) # debug
    print(f"Dataset loaded in {time.time() - start_time} seconds")
    print("length of train dataset: ", len(dataset))
    datum = dataset[0]
    print("audio slice shape: ", datum[0][0].shape)
    print("expression parameter slice shape:", datum[0][1].shape)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # exp_means, audio_means, exp_sigmas, audio_sigmas, DB_emositygen, sims, DB_labels = inference_one_epoch(args, DEE, dataloader , device, processor)
    exp_means, audio_means, exp_sigmas, audio_sigmas, DB_emositygen, sims, DB_labels = inference_one_epoch_wo_visualization(args, DEE, dataloader , device, processor) 
    print('saving outputs...')

    if args.full_length :
        length = 'full'
    else :
        length = 'short'
        
    additions = ''
    if args.random_slice:
        additions += '_random_slice'
    if args.repetition is not None :
        additions += f'_repition{args.repetition}'
    if args.statement is not None :
        additions += f'_statement{args.statement}'
    if args.key_emotions:
        additions += '_key_emotions'
        
    output_save_dir_name = f'outputs_{args.inference_method}_{epoch}_{args.split}_{args.dataset}_{length}{additions}'
    output_save_dir = os.path.join(args.save_dir, output_save_dir_name)
    os.makedirs(output_save_dir, exist_ok=True)
    
    np.save(os.path.join(output_save_dir, 'exp_means.npy'), exp_means)
    np.save(os.path.join(output_save_dir, 'audio_means.npy'), audio_means)
    np.save(os.path.join(output_save_dir, 'exp_sigmas.npy'), exp_sigmas)
    np.save(os.path.join(output_save_dir, 'audio_sigmas.npy'), audio_sigmas)
    np.save(os.path.join(output_save_dir, 'DB_labels.npy'), DB_labels)

    print('done')
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser('train', parents=[get_args_parser()])
    # args = parser.parse_args()
    parser = argparse.ArgumentParser('train DEE', add_help=False)
    parser.add_argument('--config_path', type=str, default='./config.json')
    # args will be initialize with config.json
    # and the rest will overwrite the config.json for visualization!
    parser.add_argument('--dataset', default='MEAD', type=str, help='choose dataset', choices=['RAVDESS', 'MEAD'])
    parser.add_argument('--audio_feature_dir', default=None, type=str, help='path to audio feature directory')
    parser.add_argument('--expression_feature_dir', default=None, type=str, help='path to expression feature directory')
    
    parser.add_argument('--split', default='val', type=str, help='choose which data to use at visualization')
    parser.add_argument('--num_ckpt', default=None, type=int, help='index of checkpoint for validation')
    parser.add_argument('--last_ckpt', action='store_true', help='use last checkpoint for validation')
    parser.add_argument('--best_ckpt', action='store_true', help='use best checkpoint for validation')
    
    parser.add_argument('--random_slice', action='store_true', help='randomly slice audio and expression')
    parser.add_argument('--full_length', action='store_true', help='use full length of audio and expression')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for validation')
    
    parser.add_argument('--repetition', default=None, type=int, help='choose what repetition to use for RAVDESS dataset, if None, use all')
    parser.add_argument('--statement', default=None, type=int, help='choose what statement to use for RAVDESS dataset, if None, use all')
    parser.add_argument('--key_emotions', action='store_true', help='use calm, happy, sad, angry emotions for visualization')
    
    
    args = parser.parse_args()
    args_from_json = json2args(args.config_path)
    
    # overwrite args_from_json with args
    args_dict = vars(args)
    for key in args_dict.keys():
        args_from_json.__setattr__(key, args_dict[key])
    
    print("setting use_embeddings to False as when inference, we don't need to initialize the model with emo2vec!!")
    args_from_json.use_embeddings = False
    
    print(args_from_json)
    
    print("="*25 + "infernece starts" + "="*25)
    inference(args_from_json)
    print("DONE!")

