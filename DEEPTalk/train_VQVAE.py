import argparse
import logging
import os, random
import sys

import json
import numpy as np
import torch
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "600"

from datasets_ import dataset

from models import VAEs
from models.flame_models import flame
from utils.extra import seed_everything
from utils.loss_ import calc_vq_flame_L1_loss, calc_vq_flame_L2_loss, calc_vq_vertice_L2_loss, calculate_vertex_velocity_loss, calculate_flame_jaw_loss
from utils.lr_utils import WarmupConstantSchedule
def train_one_epoch(config, epoch, model, FLAME, optimizer,scheduler, data_loader, device):
    """
    Train the model for one epoch
    """
    model.train()
    model.to(device)
    loss_epoch = 0
    total_steps = len(data_loader)
    velocity_loss = None
    for i, data in enumerate(data_loader):
        exp_param = data[:,:,:50].to(device)
        jaw_pose = data[:,:,50:53].to(device)
        inputs = torch.cat([exp_param, jaw_pose], dim=-1)
        prediction, quant_loss, info = model(inputs)
        
        if config["training"]["loss"] == "flame_L1":
            loss, recon_loss = calc_vq_flame_L1_loss(prediction, inputs, quant_loss)
        elif config["training"]["loss"] == "flame_L2":
            loss,recon_loss = calc_vq_flame_L2_loss(prediction, inputs, quant_loss)
        elif config["training"]["loss"] == "vertice_L2":
            exp_param_pred = prediction[:,:,:50].to(device)
            jaw_pose_pred = prediction[:,:,50:53].to(device)
            vertices_pred = flame.get_vertices_from_flame(config, FLAME, exp_param_pred, jaw_pose_pred, device)
            vertices_target = flame.get_vertices_from_flame(config, FLAME, exp_param, jaw_pose, device)
            loss, recon_loss = calc_vq_vertice_L2_loss(vertices_pred, vertices_target, quant_loss,
                                                       config["training"]["quant_loss_wight"], 
                                                       config["training"]["recon_loss_weight"])
        else:
            raise ValueError("Invalid loss function")
        if config["training"]["velocity_loss"]:
            velocity_loss = calculate_vertex_velocity_loss(vertices_pred, vertices_target) * config["training"]["velocity_loss_weight"]
            loss += velocity_loss
            wandb.log({"train vel loss (step)": velocity_loss.detach().item()})
        if config["training"]["flame_jaw_loss"]:
            flame_jaw_loss = calculate_flame_jaw_loss(prediction, inputs) * config["training"]["flame_jaw_loss_weight"]
            loss += flame_jaw_loss
            wandb.log({"train flame jaw loss (step)": flame_jaw_loss.detach().item()})
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        if config["training"]["lr_schedule"] == "warmup":
            scheduler.step()
        loss_epoch += loss.detach().item()
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
        if velocity_loss is not None:
            wandb.log({"train vel loss (step)": velocity_loss.detach().item()})
        wandb.log({"train loss (step)": loss.detach().item()})
        wandb.log({"train recon loss (step)": recon_loss.detach().item()})
        wandb.log({"train quant loss (step)": quant_loss.mean().detach().item()})
        if config.get("VQuantizer") and config["VQuantizer"]["reset_unused_codes"]:
            # batch index count (use non-deterministic for this operation)
            used_indices = info[2]
            torch.use_deterministic_algorithms(False,warn_only=True)
            used_indices = torch.bincount(used_indices.view(-1), minlength=model.quantize.num_embeddings)
            torch.use_deterministic_algorithms(True,warn_only=True)
            model.train_epoch_usage_count = used_indices if model.train_epoch_usage_count is None else + used_indices
    if config.get("VQuantizer") and config["VQuantizer"]["reset_unused_codes"]:      
        if (epoch % config["VQuantizer"]["reinit_every_n_epochs"] == 0 and epoch > 0):
            model.quantize.reinit_unused_codes(model.quantize.get_codebook_usage(model.train_epoch_usage_count)[0])
            model.train_epoch_usage_count = None
    if config["training"]["lr_schedule"] == "cosine":
        scheduler.step()
        
    wandb.log({"train loss (epoch)": loss_epoch / total_steps})
    print("Train Epoch: {}\tAverage Loss: {:.6f}".format(epoch, loss_epoch / total_steps))
        
def val_one_epoch(config, epoch, model, FLAME, data_loader, device):
    model.eval()
    model.to(device)
    loss_epoch = 0
    flame_recon_loss_epoch = 0
    vertices_recon_loss_epoch = 0
    quant_loss_epoch = 0
    total_steps = len(data_loader)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            exp_param = data[:,:,:50].to(device)
            jaw_pose = data[:,:,50:53].to(device)
            inputs = torch.cat([exp_param, jaw_pose], dim=-1)

            prediction, quant_loss, info = model(inputs)
            exp_param_pred = prediction[:,:,:50].to(device)
            jaw_pose_pred = prediction[:,:,50:53].to(device)
            vertices_pred = flame.get_vertices_from_flame(config, FLAME, exp_param_pred, jaw_pose_pred, device)
            vertices_target = flame.get_vertices_from_flame(config, FLAME, exp_param, jaw_pose, device)
            loss, vertices_recon_loss = calc_vq_vertice_L2_loss(vertices_pred, vertices_target, quant_loss,
                                                       config["training"]["quant_loss_wight"], 
                                                       config["training"]["recon_loss_weight"])
            if config["training"]["loss"] == "flame_L1":
                loss, flame_recon_loss = calc_vq_flame_L1_loss(prediction, inputs, quant_loss)
                flame_recon_loss_epoch += flame_recon_loss.detach().item()
            elif config["training"]["loss"] == "flame_L2":
                loss,flame_recon_loss = calc_vq_flame_L2_loss(prediction, inputs, quant_loss)
                flame_recon_loss_epoch += flame_recon_loss.detach().item()
            elif config["training"]["loss"] == "vertice_L2":
                loss = loss
            else:
                raise ValueError("Invalid loss function")
            
            vertices_recon_loss_epoch += vertices_recon_loss.detach().item()
            quant_loss_epoch += quant_loss.mean().detach().item()
            loss_epoch += loss.detach().item()
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
        wandb.log({'val flame recon loss': flame_recon_loss_epoch / total_steps})
        wandb.log({'val vertices recon loss': vertices_recon_loss_epoch / total_steps})
        wandb.log({"val quant loss": quant_loss_epoch / total_steps})
        print("Val Epoch: {}\tAverage Loss: {:.6f}".format(epoch, avg_loss))
    return avg_loss, vertices_recon_loss_epoch / total_steps

def main(args):
    """training loop for VQVAE (FLINT) in EMOTE
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # use cpu for now
    print('using device', device)
    
    seed_everything(42)

    # models
    print("Loading Models...")
    if args.model == "VQVAE":
        VQVAE = VAEs.VQVAE(config, version="v1")
        if VQVAE.quantize.__class__.__name__ == "EMAVectorQuantizer":
            VQVAE.quantize.init_codebook()
    elif args.model == "VQVAE2":
        VQVAE = VAEs.VQVAE2(config)
        if VQVAE.quantize_t.__class__.__name__ == "EMAVectorQuantizer":
            VQVAE.quantize_t.init_codebook()
        if VQVAE.quantize_b.__class__.__name__ == "EMAVectorQuantizer":
            VQVAE.quantize_b.init_codebook()
        
    FLAME_train = flame.FLAME(config, batch_size=config["training"]["batch_size"]).to(device).eval()
    FLAME_val = flame.FLAME(config, batch_size=config["validation"]["batch_size"]).to(device).eval()
    FLAME_train.requires_grad_(False)
    FLAME_val.requires_grad_(False)

    print("Loading Dataset...")
    train_dataset = dataset.MotionPriorMEADDataset(config['data'],split='train')
    val_dataset = dataset.MotionPriorMEADDataset(config['data'],split='val')
    print('val_dataset', len(val_dataset),'| train_dataset', len(train_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["validation"]["batch_size"], shuffle=True,drop_last=True) # this may not be optimal

    optimizer = torch.optim.AdamW(VQVAE.parameters(), lr=config["training"]["lr"])
    if config["training"]["lr_schedule"] == "warmup":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=config["training"]["warmup_steps"])
    elif config["training"]["lr_schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])
    save_dir = os.path.join(config["training"]["save_dir"], config["name"])
    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = 1000000
    for epoch in range(0, config["training"]['num_epochs']):
        print('epoch', epoch, 'num_epochs', config["training"]['num_epochs'])
        train_one_epoch(config, epoch, VQVAE, FLAME_train, optimizer, scheduler,train_dataloader, device)
        val_loss, val_recon_loss = val_one_epoch(config, epoch, VQVAE, FLAME_val, val_dataloader, device)
        print("-"*50)
        
        if val_recon_loss < best_loss:
            best_loss = val_recon_loss
            torch.save(
                VQVAE.state_dict(),
                os.path.join(
                    config["training"]["save_dir"],
                    config["name"],
                    "VQVAE_best.pth",
                ),
            )
            print("Save best model at {}\n".format(epoch))
            
        if (epoch != 0) and (epoch % config["training"]["save_step"] == 0) :
            torch.save(
                VQVAE.state_dict(),
                os.path.join(
                    config["training"]["save_dir"],
                    config["name"],
                    "VQVAE_{}_{:.10f}.pth".format(epoch, val_recon_loss),
                ),
            )
            print("Save model at {}\n".format(epoch))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, default="VQVAE", choices=["VQVAE", "VQVAE2"])
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
    
