import argparse
import logging
import os, random
import sys

import json
import numpy as np
import torch
import wandb

from datasets import dataset

from models import TVAE_inferno
from models.flame_models import flame
from utils.extra import seed_everything
from utils.loss import calc_vae_loss, calc_vae_loss_with_facemask

def train_one_epoch(config, epoch, model, FLAME, optimizer, data_loader, device):
    """
    Train the model for one epoch
    """
    model.train()
    model.to(device)
    loss_epoch = 0
    KLD_epoch = 0
    recon_loss_epoch = 0
    total_steps = len(data_loader)
    for i, data in enumerate(data_loader):
        exp_param = data[:,:,:50].to(device)
        jaw_pose = data[:,:,50:53].to(device)
        # jaw_pose = data[:,:,53:56].to(device)
        
        inputs = torch.cat([exp_param, jaw_pose], dim=-1)
        
        params_pred, mu, logvar = model(inputs)
        exp_param_pred = params_pred[:,:,:50].to(device)
        # already made jaw param [53:26] in data_loader
        jaw_pose_pred = params_pred[:,:,50:53].to(device)
        # jaw_pose_pred = params_pred[:,:,53:56].to(device)

        vertices_pred = flame.get_vertices_from_flame(config, FLAME, exp_param_pred, jaw_pose_pred, device)
        vertices_target = flame.get_vertices_from_flame(config, FLAME, exp_param, jaw_pose, device)
        
        loss, recon_loss, KLD_loss = calc_vae_loss(vertices_pred, vertices_target, mu, logvar)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        loss_epoch += loss.detach().item()
        KLD_epoch += KLD_loss.detach().item()
        recon_loss_epoch += recon_loss.detach().item()
        if i % config["training"]["log_step"] == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    i * len(data),
                    len(data_loader.dataset),
                    100.0 * i / len(data_loader),
                    loss.item(),
                )
            )
        wandb.log({"train mean (step)": mu.mean().detach().item()})
        wandb.log({"train std (step)": logvar.mean().detach().item()})
        wandb.log({"train recon loss (step)": recon_loss.detach().item()})
        wandb.log({"train KLD (step)": KLD_loss.detach().item()})
        wandb.log({"train loss (step)": loss.detach().item()})
    wandb.log({"train loss (epoch)": loss_epoch / total_steps})
    wandb.log({"train recon loss (epoch)": recon_loss_epoch / total_steps})
    wandb.log({"train KLD (epoch)": KLD_epoch / total_steps})
    print("Train Epoch: {}\tAverage Loss: {:.6f}".format(epoch, loss_epoch / total_steps))
        
def val_one_epoch(config, epoch, model, FLAME, data_loader, device):
    model.eval()
    model.to(device)
    loss_epoch = 0
    KLD_epoch = 0
    recon_loss_epoch = 0
    total_steps = len(data_loader)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            exp_param = data[:,:,:50].to(device)
            jaw_pose = data[:,:,50:53].to(device)
            # jaw_pose = data[:,:,53:56].to(device)
            
            inputs = torch.cat([exp_param, jaw_pose], dim=-1)
            
            params_pred, mu, logvar = model(inputs)
            exp_param_pred = params_pred[:,:,:50].to(device)
            jaw_pose_pred = params_pred[:,:,50:53].to(device)
            # jaw_pose_pred = params_pred[:,:,53:56].to(device)

            vertices_pred = flame.get_vertices_from_flame(config, FLAME, exp_param_pred, jaw_pose_pred, device)
            vertices_target = flame.get_vertices_from_flame(config, FLAME, exp_param, jaw_pose, device)
            
            loss, recon_loss, KLD_loss = calc_vae_loss(vertices_pred, vertices_target, mu, logvar)
            
            loss_epoch += loss.detach().item()
            KLD_epoch += KLD_loss.detach().item()
            recon_loss_epoch += recon_loss.detach().item()
            if i % config["training"]["log_step"] == 0:
                print(
                    "Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        i * len(data),
                        len(data_loader.dataset),
                        100.0 * i / len(data_loader),
                        loss.item(),
                    )
                )
        avg_loss = loss_epoch / total_steps
        wandb.log({"val loss": avg_loss})
        wandb.log({"val recon loss": recon_loss_epoch / total_steps})
        wandb.log({"val KLD loss": KLD_epoch / total_steps})
        print("Val Epoch: {}\tAverage Loss: {:.6f}".format(epoch, avg_loss))
    return avg_loss

def main(args):
    """training loop for TVAE (FLINT) in EMOTE
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # use cpu for now
    print('using device', device)
    
    seed_everything(42)
      
    # models
    print("Loading Models...")
    TVAE = TVAE_inferno.TVAE(config)
    
    FLAME_train = flame.FLAME(config, batch_size=config["training"]["batch_size"]).to(device).eval()
    FLAME_val = flame.FLAME(config, batch_size=config["validation"]["batch_size"]).to(device).eval()
    FLAME_train.requires_grad_(False)
    FLAME_val.requires_grad_(False)

    print("Loading Dataset...")
    train_dataset = dataset.MotionPriorMEADDataset(config['data'],split='train')
    # train_dataset = dataset.MotionPriorMEADDataset(config['data'],split='debug')
    val_dataset = dataset.MotionPriorMEADDataset(config['data'],split='val')
    # val_dataset = dataset.MotionPriorMEADDataset(config['data'],split='debug')
    print('val_dataset', len(val_dataset),'| train_dataset', len(train_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["validation"]["batch_size"], drop_last=True) # this may not be optimal
    
    optimizer = torch.optim.Adam(TVAE.parameters(), lr=config["training"]["lr"])
    save_dir = os.path.join(config["training"]["save_dir"], config["name"])
    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = 1000000
    for epoch in range(0, config["training"]['num_epochs']):
        print('epoch', epoch, 'num_epochs', config["training"]['num_epochs'])
        train_one_epoch(config, epoch, TVAE, FLAME_train, optimizer, train_dataloader, device)
        val_loss = val_one_epoch(config, epoch, TVAE, FLAME_val, val_dataloader, device)
        print("-"*50)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                TVAE.state_dict(),
                os.path.join(
                    config["training"]["save_dir"],
                    config["name"],
                    "TVAE_best.pth",
                ),
            )
            print("Save best model at {}\n".format(epoch))
            
        if (epoch != 0) and (epoch % config["training"]["save_step"] == 0) :
            torch.save(
                TVAE.state_dict(),
                os.path.join(
                    config["training"]["save_dir"],
                    config["name"],
                    "TVAE_{}_{:.10f}.pth".format(epoch, val_loss),
                ),
            )
            print("Save model at {}\n".format(epoch))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    print(args)
    
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)
      
    print("saving config to", os.path.join(config["training"]["save_dir"], config["name"]))
    os.makedirs(os.path.join(config["training"]["save_dir"], config["name"]), exist_ok=True)
    with open(os.path.join(config["training"]["save_dir"], config["name"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        

    wandb.init(project = config["project_name"],
            name = config["name"],
            config = config)
    
    main(args)
    
