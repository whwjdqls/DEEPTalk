import torch
import yaml
import argparse
import wandb
from omegaconf import OmegaConf, DictConfig
import time
import sys
import os
import numpy as np
os.environ["WANDB__SERVICE_WAIT"] = "300" 
sys.path.append('../')
from DEE.utils.utils import seed_everything, generate_date_time
from FER.models.MLP import MLP
from FER.datasets.affectnet_dataset import AffectNetExpressionDataset

def get_yaml_config(yaml_file):
    config = OmegaConf.load(yaml_file)
    return config


def get_accuracy(preds, labels):
    """
    Calculate the accuracy of the predicted emotion classes.
    Args:
        preds (numpy.ndarray): Array of predicted emotion classes.
        labels (numpy.ndarray): Array of true emotion classes.
    Returns:
        tuple: A tuple containing the class-wise accuracy and the overall accuracy.
    """
    emotion_classes = np.unique(labels)
    check_correct = preds == labels
    class_num = [len(labels[labels == emotion_class]) for emotion_class in emotion_classes]
    class_correct = [0. for _ in range(len(emotion_classes))]
    for emotion_class in emotion_classes:
        class_correct[emotion_class] = check_correct[labels == emotion_class].sum().item()
    assert sum(class_correct) == check_correct.sum().item()
    class_acc = [class_correct[i] / class_num[i] for i in emotion_classes]
    total_acc = check_correct.sum().item() / len(labels)
    return class_acc, total_acc

def augment_expression_data(data, instance_prob=0.5, std=0.01):
    """
    Augments the expression data by adding perturbations to a subset of instances.
    Args:
        data (torch.Tensor): The input data tensor.
        instance_prob (float, optional): The probability of perturbing each instance. Defaults to 0.5.
        std (float, optional): The standard deviation of the perturbations. Defaults to 0.01.
    Returns:
        torch.Tensor: The augmented data tensor.
    """
    if instance_prob == 0 or std == 0:
        return data
    mean = 0
    parameter_prob = 0.7
    mask = torch.rand_like(data) < parameter_prob
    perturbations = torch.normal(mean=mean, std=std, size=data.size()).to(data.device)
    indices = torch.rand(data.size(0)) < instance_prob
    data[indices] += perturbations[indices] * mask[indices]
    return data

def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch,
                    use_jaw=True, aug_prob=0, aug_std = 0, enable_wandb=False):
    model.train()
    model.to(device)
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs_ = inputs[:,:50]
        if use_jaw:
            inputs_ = torch.cat((inputs_, inputs[:,53:56]), dim = 1)
        inputs = augment_expression_data(inputs_,instance_prob=aug_prob, std=aug_std)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # outputs -> exp ,  labels -> int : crossentropy only
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'epoch [{epoch}] [{int(i/len(data_loader)*100)}%] loss : {loss.item():.5f}') # format loss to show only 3 decimal points
        if enable_wandb:
            wandb.log({'train loss step':loss.item()})
        running_loss += loss.item()
    avg_loss = running_loss / len(data_loader)
    print(f'epoch [{epoch}] loss : {avg_loss:.5f}')
    if enable_wandb:
        wandb.log({'train loss epoch' : avg_loss})
        
@torch.no_grad()
def val_one_epoch(model, data_loader, criterion, optimizer, device, epoch, use_jaw=True, enable_wandb=True):
    model.eval()
    model.to(device)
    running_loss = 0.0
    preds_list = []
    labels_list = []
    for inputs, labels in data_loader:
        inputs_ = inputs[:,:50]
        if use_jaw:
            inputs_ = torch.cat((inputs_, inputs[:,53:56]), dim = 1)
        inputs = inputs_
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        preds = outputs.argmax(1)
        preds_list.append(preds.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        # preds = outputs.argmax(1)
        # acc_dict = get_accuracy_for_each_class(preds, labels)
        # num_correct += (preds == labels).sum().item()
        
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    preds_list = np.concatenate(preds_list)
    labels_list = np.concatenate(labels_list)
    num_correct = (preds_list == labels_list).sum().item()
    class_acc, total_acc = get_accuracy(preds_list, labels_list)
    
    avg_loss = running_loss / len(data_loader)
    print(f'epoch [{epoch}] val loss : {avg_loss:.5f}')
    print(f'epoch [{epoch}] val acc : {total_acc:.3f}')
    if enable_wandb:
        wandb.log({'val loss epoch' : avg_loss})
        wandb.log({'val acc epoch' : total_acc})
        for i, acc in enumerate(class_acc):
            wandb.log({f'class acc/{i}':acc})
    return avg_loss, total_acc
    
    
def main(args):
    cfg = get_yaml_config(args.config)
    if args.wandb:
        wandb.init(project=f'FER_{cfg.model.output_dim}', name=cfg.utils.exp_name, config=dict(cfg))
    date_time = generate_date_time()
    save_path = cfg.train.save_dir + '/' + cfg.utils.exp_name + date_time
    os.makedirs(save_path, exist_ok=True)
    # save config file
    with open(f'{save_path}/config.yaml','w') as fp:
        OmegaConf.save(config=cfg, f=fp.name)
    seed_everything(cfg.utils.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if cfg.model.name == 'MLP':
        model = MLP(input_dim = cfg.model.input_dim, # if cfg.model.input_dim is 53, we are using jaw pose
                    layers = cfg.model.layers, 
                    output_dim = cfg.model.output_dim, # cfg.model.output_dim is same as label num
                    dropout = cfg.model.dropout, 
                    batch_norm = cfg.model.batch_norm, 
                    activation = cfg.model.activation)
    else: 
        raise NotImplementedErrors
    # dataset
    train_dataset = AffectNetExpressionDataset(data_file_path=cfg.data.data_file_path,
                                            label_file_path=cfg.data.label_file_path,
                                            exclude_7=cfg.data.exclude_7,
                                            exception_file_path=cfg.data.exception_file_path,
                                            split='train')
    val_dataset = AffectNetExpressionDataset(data_file_path=cfg.data.data_file_path,
                                            label_file_path=cfg.data.label_file_path,
                                            exclude_7=cfg.data.exclude_7,
                                            split='val')
    
    print("train dataset length : ", len(train_dataset))
    print("val dataset length : ", len(val_dataset))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val.batch_size, shuffle=False)

    if cfg.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    else:
        raise NotImplementedError('Not implemented optimizer')

    if cfg.train.loss == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg.train.loss == 'WeightedCrossEntropyLoss':
        class_count = np.array([ 74826., 134292.,  25444.,  14078.,   6374.,   3800.,  24870.,   3746.])
        if cfg.model.output_dim == 7:
            class_count = class_count[:-1]
        class_weight = sum(class_count) / (len(class_count) * class_count)
        class_weight = torch.tensor(class_weight, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight,reduction='mean')
    else:
        raise NotImplementedError('Not implemented loss')
    if cfg.train.scheduler == 'CosineAnnealingLR':
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs, eta_min=0.000001)
          
    else:
        raise NotImplementedError('Not implemented scheduler')
    use_jaw = True
    if cfg.model.input_dim == 50:
        use_jaw = False
        
    best_val_loss = float('inf')
    for epoch in range(cfg.train.num_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                        use_jaw=use_jaw, 
                        aug_prob=cfg.train.data_aug.prob, aug_std=cfg.train.data_aug.std,
                        enable_wandb=args.wandb)
        scheduler.step()
        
        val_loss, accuracy = val_one_epoch(model, val_loader, criterion, optimizer, device, epoch, use_jaw=use_jaw, enable_wandb=args.wandb)
        
        if (epoch%(cfg.train.save_every) == 0) and (epoch != 0):
            torch.save(model.state_dict(), f'{save_path}/model_{epoch}_acc_{accuracy:.2f}_val_{val_loss:.5f}.pth')
            print('model saved')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_path}/model_best.pth')
            print('model saved')
        
    

                              
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    args = parser.parse_args()
    
    main(args)
    
    
    
