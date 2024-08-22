import torch
from omegaconf import OmegaConf, DictConfig
import sys
import os

sys.path.append(f'../')
from FER.models.MLP import MLP
from DEE.utils.utils import compare_checkpoint_model

def get_yaml_config(yaml_file):
    config = OmegaConf.load(yaml_file)
    return config


def init_affectnet_feature_extractor(config_path, model_path):
    cfg = get_yaml_config(config_path)
    model = None
    if cfg.model.name == 'MLP':
        model = MLP(input_dim = cfg.model.input_dim, # if cfg.model.input_dim is 53, we are using jaw pose
                    layers = cfg.model.layers, 
                    output_dim = cfg.model.output_dim, # cfg.model.output_dim is same as label num
                    dropout = cfg.model.dropout, 
                    batch_norm = cfg.model.batch_norm, 
                    activation = cfg.model.activation)
    else: 
        raise NotImplementedErrors
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    compare_checkpoint_model(checkpoint, model)
    return cfg, model

if __name__ == '__main__':
    config_path = './checkpoint/config.yaml'
    model_path = './checkpoint/model_best.pth'
    cfg, model = init_affectnet_feature_extractor(config_path, model_path)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    inputs = torch.rand(10, 53).to(device)
    outputs = model(inputs)
    print(outputs.shape)
    
    features = model.extract_feature_from_layer(inputs, -2)
    print(features.shape)
    
    print('done')