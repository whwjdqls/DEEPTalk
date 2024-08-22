import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Config, Wav2Vec2Processor, AutoConfig
from .wav2vec import Wav2Vec2Model

import torch.nn.functional as F
import copy
import einops

import fairseq 
from dataclasses import dataclass

from .gradient_reversal import GradientReversal
from .aggr import GPO, AvgPool
import sys
sys.path.append('../')
from FER.get_model import init_model_from_path


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Compute the positional encodings in advance
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros((max_seq_len, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1)].clone().detach()
        return self.dropout(x)
    
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

  
@dataclass
class UserDirModule:
    user_dir: str
    

class PointAudioEncoder(nn.Module): 
    def __init__(self, args) :
        super(PointAudioEncoder, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'

        model_dir = './DEE/models/emo2vec'
        model_path = UserDirModule(model_dir)
        emo2vec_checkpoint = './DEE/models/emo2vec/emotion2vec_base.pt'
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([emo2vec_checkpoint])
        self.embedding_model = model[0]

        if self.freeze_embedding_model:
            for param in self.embedding_model.parameters():
                param.requires_grad = False
            self.embedding_model.eval()
            
        if args.num_audio_layers == 0:
            self.audio_encoder = nn.Identity()
        else:
            self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=args.num_audio_heads, batch_first=True)
            self.audio_encoder = nn.TransformerEncoder(self.audio_encoder_layer, num_layers=args.num_audio_layers)
        
        self.feature_map = nn.Linear(768, args.feature_dim)
        
        self.audio_pool = args.audio_pool
        if self.audio_pool == 'gpo':
            self.audio_aggr = GPO(32,32)
        elif self.audio_poool == 'cls':
            self.audio_cls_token = nn.Parameter(torch.rand(1,1,768)) # just initialize it for the sake of compatibility
        
    def get_feature(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        self.embedding_model.eval() # make sure embedding model is in eval mode
        embeddings = self.embedding_model.extract_features(audio, padding_mask=None) 
        embeddings = embeddings['x'] # (BS, 50*sec, 768)
        
       if self.audio_pool == 'cls':
            batch_size = embeddings.shape[0]
            cls_token = einops.repeat(self.audio_cls_token, '() n e -> b n e', b=batch_size).to(self.device)
            embeddings = torch.cat([cls_token, embeddings], dim=1) # (BS,13,256)
            
        output = self.audio_encoder(embeddings) # (BS, 50*sec, 768)
        
        if self.audio_pool == 'avg':
            output = torch.mean(output, dim=1)
        elif self.audio_pool == 'max':
            output = torch.max(output, dim=1).values
        elif self.audio_pool == 'gpo':
            output,_ = self.audio_aggr(output)
        elif self.audio_pool == 'cls':
            output = output[:,0,:]
        return output
    
    def forward(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        audio_emb = self.get_feature(audio)
        output = self.audio_feature_map(audio_emb)
        return output

class PointExpEncoder(nn.Module): 
    def __init__(self, args) :
        super(PointExpEncoder, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.exp_max_seq_len = args.max_seq_len * 30
        self.pos_encoding = args.pos_encoding
        ## For parameter encoder
        self.exp_map = nn.Linear(args.parameter_dim, args.parameter_feature_dim)
        if self.pos_encoding == 'sinusoidal' :
            self.exp_pos_encoder = PositionalEncoding(args.parameter_feature_dim, self.exp_max_seq_len)
        elif self.pos_encoding == 'alibi' :
            self.exp_mask = init_alibi_biased_mask_future(args.num_parameter_heads, self.exp_max_seq_len)
            
        self.exp_encoder_layer = nn.TransformerEncoderLayer(d_model=args.parameter_feature_dim, nhead=args.num_parameter_heads, batch_first=True)
        self.exp_encoder = nn.TransformerEncoder(self.exp_encoder_layer, num_layers=args.num_parameter_layers)
        
        self.parameter_feature_map = nn.Linear(args.parameter_feature_dim, args.feature_dim)
        self.exp_pool = args.exp_pool
        if self.exp_pool == 'gpo':
            self.exp_aggr = GPO(32,32)
        elif self.exp_pool == 'cls':
            self.parameter_cls_token = nn.Parameter(torch.rand(1,1,args.parameter_feature_dim))
        
        
    def get_feature(self, raw_exp) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        exp = self.exp_map(raw_exp) # (BS, seq_len, model_dim)
        
       if self.exp_pool == 'cls':
            batch_size = exp.shape[0]
            cls_token = einops.repeat(self.parameter_cls_token, '() n e -> b n e', b=batch_size).to(self.device)
            embeddings = torch.cat([cls_token, exp], dim=1) # (BS,13,256)
        
        mask = None
        if self.pos_encoding == 'sinusoidal' :
            exp = self.exp_pos_encoder(exp)# [BS, seq_len, 768]
        elif self.pos_encoding == 'alibi' :
            B,T = exp.shape[:2] # B,T
            mask = self.exp_mask[:, :T, :T].clone() \
                .detach().to(device=exp.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)

        output = self.exp_encoder(exp, mask=mask) 
        
        if self.exp_pool == 'avg':
            output = torch.mean(output, dim=1)
        elif self.exp_pool == 'max':
            output = torch.max(output, dim=1).values
        elif self.exp_pool == 'gpo':
            output,_ = self.exp_aggr(output)
        elif self.exp_pool == 'cls':
            output = output[:,0,:]
            
    def forward(self, raw_exp) :
        exp_emb = self.get_feature(raw_exp)
        output = self.parameter_feature_map(exp_emb)
        return output
    
class PointDEE(nn.Module):
    def __init__(self, args):
        super(ProbDEE, self).__init__()
        self.audio_encoder = PointAudioEncoder(args)
        self.exp_encoder = PointExpEncoder(args)

    def forward(self, audio, raw_exp):
        audio_emb = self.audio_encoder(audio)
        exp_emb = self.exp_encoder(raw_exp)
        
        audio_emb = l2norm(audio_emb, dim=-1)
        exp_emb = l2norm(exp_emb, dim=-1)
        
        return audio_emb, exp_emb
    
