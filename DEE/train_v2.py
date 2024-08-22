import os
import torch
os.environ["WANDB__SERVICE_WAIT"] = "300"
import argparse
from config import get_args_parser
import models
import datasets
from utils.loss import CLIP_loss, RECO_loss, emotion_guided_loss_gt, CLIP_loss_with_expression_guide
from torch.utils.data import DataLoader
from utils.utils import seed_everything
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
from transformers import Wav2Vec2Processor
torch.cuda.empty_cache()
import sys
sys.path.append('../')
from RER.get_model import init_affectnet_feature_extractor


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training start using {device}...")
    
    print("Loading models...")
    DEE = models.PointDEE(args)
    DEE = DEE.to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    if args.use_speaker_norm:
        SID = models.SIDetector(args)
        SID = SID.to(device)
    config_path = '../FER/checkpoint/config.yaml'
    model_path = '../FER/checkpointmodel_best.pth'
    cfg, affectnet_feature_extractor = init_affectnet_feature_extractor(config_path,model_path)
    assert cfg.model.layers[-2] == args.parameter_dim, f"cfg.model.layers[-2] : {cfg.model.layers[-2]} != args.parameter_dim : {args.parameter_dim}"
    affectnet_feature_extractor = affectnet_feature_extractor.to(device)
    affectnet_feature_extractor.requires_grad_(False)
    
    print("Loading dataset...")
    val_dataset = datasets.AudioExpressionDataset(args, dataset='MEAD', split = 'val')
    start_time = time.time()
    train_dataset = datasets.AudioExpressionDataset(args, dataset=args.dataset, split = 'train') # debug
    print(f"Dataset loaded in {time.time() - start_time} seconds")
    print("length of train dataset: ", len(train_dataset))
    datum = train_dataset[0]
    print("audio slice shape: ", datum[0][0].shape)
    print("expression parameter slice shape:", datum[0][1].shape)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(DEE.parameters(), lr=args.lr)
    if args.loss == 'infoNCE':
        criterion = CLIP_loss
    elif args.loss == 'RECO':
        criterion = RECO_loss
    elif args.loss == 'emotion_guided_loss_gt':
        criterion = emotion_guided_loss_gt
    elif args.loss == 'CLIP_loss_with_expression_guide':
        criterion = CLIP_loss_with_expression_guide
        
    if args.use_speaker_norm:
        optimizer_SID = torch.optim.Adam(SID.parameters(), lr=args.lr)
    criterion_SID = torch.nn.CrossEntropyLoss()
    
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.epochs//3, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)
    elif args.scheduler == 'cosine_warmup':
        optimizer = torch.optim.Adam(DEE.parameters(), lr=0)
        scheduler = utils.CosineAnnealingWarmUpRestarts(optimizer,
                                                        T_0=int(args.epochs/5.), # first cycle length
                                                        T_mult=1, # cycle length multiplier
                                                        eta_max=args.lr,  #lr max learning rate
                                                        T_up=1, # warm up epochs
                                                        gamma=0.5) # cycle lr decay
    print("Training loop...")

    DEE.train()
    if args.use_speaker_norm:
        SID.train()
    if args.use_speaker_norm and args.SID_first:
        # Phase 1: Train SID_layer for the first args.SID_epochs
        # ---Note that we are NOT Training SID with adversarial loss---
        # ---we are training SID so that SID can classify well with current audio embeddings---
        for epoch in range(args.SID_epochs):
            step = 0
            for samples, labels in tqdm.tqdm(train_dataloader):
                audio_samples = samples[0].to(device)
                expression_samples = samples[1].to(device)
                expression_samples = affectnet_feature_extractor.extract_feature_from_layer(expression_samples, -2)
                audio_embedding, expression_embedding = DEE(audio_samples, expression_samples)

                pred_SID = SID(audio_embedding)
                if args.gender_norm :
                    target = labels[2] # gender
                else :
                    target = labels[3] # actor
                    
                # we are NOT adversially training SID
                loss_SID = criterion_SID(pred_SID, target.to(device)) * args.SID_lambda
                
                optimizer_SID.zero_grad() 
                loss_SID.backward() 
                optimizer_SID.step() # only update SID
                
                accuracy = evaluation.classification_accuracy(pred_SID, target.to(device))
                wandb.log({"SID/GD accuracy (Phase 1)": accuracy})
                wandb.log({"SID/GD loss (Phase 1)": loss_SID})
                if step % 50 == 0:
                    print(f"Phase 1 - Epoch {epoch}, step {step} SID loss: {loss_SID}")
                step += 1
            # if (epoch % (args.val_freq+1) == args.val_freq) and epoch != 0:
            #     torch.save(DEE.state_dict(), f"{args.save_dir}/model_{epoch}.pt")
            #     print(f"Model saved at {args.save_dir}/model_{epoch}.pt")
    else :
        args.SID_epochs = 0

    # Phase 2: Train the whole DEE for the remaining epochs
    best_val_acc = 0
    for epoch in range(args.SID_epochs, args.epochs):
        DEE.train()
        cumulative_loss = 0
        step = 0
        # train for one epoch
        for samples, labels in tqdm.tqdm(train_dataloader):
            '''
            samples : [audio_processed,expression_processed]
            labels : [emotion, intensity, gender, actor_id] ->[ int, int, int, int]
            '''
            audio_samples = processor(samples[0],sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
            expression_samples = samples[1].to(device)
            expression_samples = affectnet_feature_extractor.extract_feature_from_layer(expression_samples, -2)
            audio_embedding, expression_embedding = DEE(audio_samples, expression_samples)

            if args.loss == 'emotion_guided_loss_gt':
                loss = criterion(audio_embedding, expression_embedding, labels[0],args.gt_guide_weight, DEE.logit_scale, device)
            elif args.loss == 'CLIP_loss_with_expression_guide':
                loss = criterion(audio_embedding, expression_embedding, args.exp_guide_weight, DEE.logit_scale, device)
            else:
                loss = criterion(audio_embedding, expression_embedding, DEE.logit_scale, device)
            
            
            wandb.log({"train loss": loss})
            if args.use_speaker_norm:
                if args.gender_norm :
                    target = labels[2] # gender
                else :
                    target = labels[3] # actor

                pred_SID = SID(audio_embedding) 
                # we are NOT adversially training SID
                loss_SID = criterion_SID(pred_SID,target.to(device)) 
                accuracy = evaluation.classification_accuracy(pred_SID, target.to(device))
                wandb.log({"SID accuracy": accuracy})
                wandb.log({"SID loss": loss_SID}) # we are logging unweighted SID loss 
                
                # we do not update SID here because gradients from loss_SID should flow to DEE first
                
                loss = loss - loss_SID * args.SID_lambda # for total loss, SID loss weight is added
                
            # log/print etc
            wandb.log({"learning rate": optimizer.param_groups[0]['lr']})
            wandb.log({"logit": torch.clip(DEE.logit_scale.exp(), max = 100) })
            if step % 50 == 0:
                print(f"epoch {epoch}, step {step} loss : {loss}")
                
            step+=1

            
            if args.use_speaker_norm: # only update SID 
                optimizer_SID.zero_grad() 
                loss_SID.backward(retain_graph=True) # as we are updating SID and DEE at the same time, we need to retain graph 
            
            optimizer.zero_grad()
            loss.backward() # this loss contatins Adversarial loss for SID and CLIP loss for DEE
            
            if args.use_speaker_norm:
                optimizer_SID.step() # only update SID 
                
            optimizer.step() # only update DEE
            
            audio_samples = audio_samples.detach().cpu()
            expression_samples = expression_samples.detach().cpu()
            cumulative_loss += loss.item() # culminative loss is only for CLIP loss
            del samples
            del audio_samples
            del expression_samples
            torch.cuda.empty_cache()
                

        scheduler.step()

        batch_num_per_epoch = len(train_dataset) / args.batch_size
        print(f"Epoch {epoch} loss: {cumulative_loss/batch_num_per_epoch}") # train_dataloader
        
        # validate every args.val_freq epochs -> best recommended to 1 to validate every epoch
        if (epoch % (args.val_freq) == 0) and epoch != 0:
            # when validating while trining, don't visualize heat map
            expression_accuracy, audio_accuracy, val_loss = evaluation.validation(args, DEE, val_dataloader, visualize = False, device = device)
            wandb.log({"expression accuracy loss": expression_accuracy,
                       "audio accuracy loss": audio_accuracy})
            wandb.log({"val loss": val_loss})
            mean_acc = (audio_accuracy + expression_accuracy)/2.
            # save best model
            if mean_acc > best_val_acc:
                torch.save(DEE.state_dict(), f"{args.save_dir}/model_best.pt")
                print(f"Model saved at {args.save_dir}/model_best.pt")

            torch.save(DEE.state_dict(), f"{args.save_dir}/model_{epoch}_{cumulative_loss/batch_num_per_epoch}.pt")
            print(f"Model saved at {args.save_dir}/model_{epoch}_{cumulative_loss/batch_num_per_epoch}.pt")

    ## evaluate and save at last epoch and visualize heat map
    _, _, _, expression_accuracy, audio_accuracy, matched_audio, matched_expression, val_loss = evaluation.validation(args, DEE, val_dataloader, visualize = True, device = device)
    visualize.visualize_retrieval(expression_accuracy, audio_accuracy, matched_audio, matched_expression, args)
    
    wandb.log({"expression accuracy": expression_accuracy,
                "audio accuracy": audio_accuracy})
    torch.save(DEE.state_dict(), f"{args.save_dir}/model_{epoch}.pt")
    print(f"Model saved at {args.save_dir}/model_{epoch}.pt")
    print("Traininig DONE!")


# get retrival accuraccy on the fly

def save_arguments_to_file(args, filename='arguments.json'):
    save_path = args.save_dir + '/' + filename
    with open(save_path, 'w') as file:
        json.dump(vars(args), file)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser('train', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    wandb.init(project = args.project_name,
               name = args.wandb_name,
               config = args)
    
    print(args.save_dir)
    save_arguments_to_file(args)
    
    seed_everything(24)
    train(args)

    print("DONE!")
