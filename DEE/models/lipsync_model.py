import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Config, Wav2Vec2Processor, AutoConfig
from .wav2vec import Wav2Vec2Model

import torch.nn.functional as F
import copy
import einops
# pip install fairseq 
import fairseq # 12-30) added to support emotion2vec
from dataclasses import dataclass
from .gradient_reversal import GradientReversal
from funasr import AutoModel
import sys
from .aggr import GPO, AvgPool
if 'DEE' in sys.path[0]:
    from utils.masks import init_alibi_biased_mask_future
else: 
    from DEE.utils.masks import init_alibi_biased_mask_future

# TODOs
# [] change audio model to pretrained wav2vec2 model
class lip_sync_model(nn.Module) :
    def __init__(self, args, load_checkpoint=True) :
        super(DEE_v2, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.audio_pool = args.audio_pool
        self.exp_pool = args.exp_pool

        self.audio_max_seq_len = args.max_seq_len * 50 # audio feature is 50HZ
        self.exp_max_seq_len = args.max_seq_len * 30 # expression param is 30HZ
        self.pos_encoding = args.pos_encoding # 'alibi' or 'sinusoidal' 
        self.audio_mask = None
        self.parameter_mask = None
        # compute loss
        if args.temperature_fix :
            self.logit_scale = torch.ones([]) * np.log(1/args.temperature)
        else :
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/args.temperature))

        # embedding_len = round(49*args.audio_feature_len/16000)+1

        if self.pos_encoding == 'sinusoidal' :
            self.audio_pos_encoder = PositionalEncoding(768, self.audio_max_seq_len)
        elif self.pos_encoding == 'alibi' :
            self.audio_mask = init_alibi_biased_mask_future(args.num_audio_heads, self.audio_max_seq_len)
        self.audio_cls_token = nn.Parameter(torch.rand(1,1,768)) # just initialize it for the sake of compatibility

        # Load wav2vec pretrained model for audio encoder
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.eval()

        # self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=args.num_audio_heads, batch_first=True)
        # self.audio_encoder = nn.TransformerEncoder(self.audio_encoder_layer, num_layers=args.num_audio_layers)
        self.audio_feature_map = nn.Linear(768, args.feature_dim, bias=False)

        ## For parameter encoder
        self.parameter_map = nn.Linear(args.parameter_dim, args.parameter_feature_dim)
        if self.pos_encoding == 'sinusoidal' :
            self.parameter_pos_encoder = PositionalEncoding(args.parameter_feature_dim, self.exp_max_seq_len)
        elif self.pos_encoding == 'alibi' :
            self.parameter_mask = init_alibi_biased_mask_future(args.num_parameter_heads, self.exp_max_seq_len)
        self.parameter_cls_token = nn.Parameter(torch.rand(1,1,args.parameter_feature_dim)) # just initialize it for the sake of compatibility
        self.parameter_encoder_layer = nn.TransformerEncoderLayer(d_model=args.parameter_feature_dim, nhead=4, batch_first=True)
        self.parameter_encoder = nn.TransformerEncoder(self.parameter_encoder_layer, num_layers=args.num_parameter_layers)
        self.parameter_feature_map = nn.Linear(args.parameter_feature_dim, args.feature_dim, bias=False)

        
        if self.audio_pool == 'gpo':
            self.audio_aggr = GPO(32,32)
        if self.exp_pool == 'gpo':
            self.exp_aggr = GPO(32,32)


    def encode_audio(self, audio) :

        if self.audio_pool == 'cls':
            batch_size = feats.shape[0]
            cls_token = einops.repeat(self.audio_cls_token, '() n e -> b n e', b=batch_size).to(self.device)
            feats = torch.cat([cls_token, feats], dim=1) # (BS,13,256)
        
        mask = None
        if not self.no_audio_PE:
            if self.pos_encoding == 'sinusoidal' :
                feats = self.audio_pos_encoder(feats)# [BS, seq_len, 768]
            elif self.pos_encoding == 'alibi' :
                B,T = feats.shape[:2] # B,T
                mask = self.audio_mask[:, :T, :T].clone() \
                    .detach().to(device=feats.device)
                if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                    mask = mask.repeat(B, 1, 1)
            
        output = self.audio_encoder(feats, mask=mask) # [BS, seq_len, 768]
 

        if self.audio_pool == 'avg':
            output = torch.mean(output, dim=1)
        elif self.audio_pool == 'max':
            output = torch.max(output, dim=1).values
        elif self.audio_pool == 'gpo':
            output,_ = self.audio_aggr(output)
        elif self.audio_pool == 'cls':
            output = output[:,0,:]
            
        output = self.audio_feature_map(output)
        return output


    def encode_parameter(self, raw_parameter) :
        '''
        raw_parameter : (BS, seq_len, 100) : (BS, 12, 100)
        '''
        parameter = self.parameter_map(raw_parameter) # (BS, 12, 256)
        batch_size = parameter.shape[0]
        # add cls_token
        if self.exp_pool == 'cls':
            cls_token = einops.repeat(self.parameter_cls_token, '() n e -> b n e', b=batch_size).to(self.device)
            parameter = torch.cat([cls_token, parameter], dim=1) # (BS,13,256)

        mask = None
        if self.pos_encoding == 'sinusoidal' :
            parameter = self.parameter_pos_encoder(parameter)
        elif self.pos_encoding == 'alibi' :
            B,T = parameter.shape[:2] # B,T
            mask = self.parameter_mask[:, :T, :T].clone() \
                .detach().to(device=parameter.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
  
        output = self.parameter_encoder(parameter, mask=mask) # (BS, 13, 256)
        
        if self.exp_pool == 'avg':
            output = torch.mean(output, dim=1)
        elif self.exp_pool == 'max':
            output = torch.max(output, dim=1).values
        elif self.exp_pool == 'gpo':
            output,_ = self.exp_aggr(output)
        elif self.exp_pool == 'cls':
            output = output[:,0,:] # (BS,512)  
        else:
            raise ValueError('Invalid expression pooling method') 
                 
        output = self.parameter_feature_map(output) # (BS, 13, 512)
        return output

        
    def forward(self, audio, parameter) :
        """
        audio: (batch_size, sampling_rate=16000)
        parameter: (batch_size, Expression_param=100)
        """
        audio_features = self.encode_audio(audio)
        parameter_features = self.encode_parameter(parameter)

        # normalized features
        audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)
        parameter_features = parameter_features / parameter_features.norm(dim=1, keepdim=True)

        return audio_features, parameter_features
    
    def get_audio_embedding(self, audio):
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        self.embedding_model.eval() # always set to eval mode
        feats = self.embedding_model.extract_features(audio, padding_mask=None) 
        feats = feats['x'] # (BS, 50*sec, 768)
        if self.audio_pool == 'cls':
            batch_size = feats.shape[0]
            cls_token = einops.repeat(self.audio_cls_token, '() n e -> b n e', b=batch_size).to(self.device)
            feats = torch.cat([cls_token, feats], dim=1) # (BS,13,256)
        
        mask = None
        if self.pos_encoding == 'sinusoidal' :
            feats = self.audio_pos_encoder(feats)# [BS, seq_len, 768]
        elif self.pos_encoding == 'alibi' :
            B,T = feats.shape[:2] # B,T
            mask = self.audio_mask[:, :T, :T].clone() \
                .detach().to(device=feats.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
        output = self.audio_encoder(feats, mask=mask) # [BS, seq_len, 768]
 
        if self.audio_pool == 'avg':
            output = torch.mean(output, dim=1)
        elif self.audio_pool == 'max':
            output = torch.max(output, dim=1).values
        elif self.audio_pool == 'gpo':
            output,_ = self.audio_aggr(output)
        elif self.audio_pool == 'cls':
            output = output[:,0,:]
        return output
    
    def get_parameter_embedding(self, raw_parameter) :
        '''
        raw_parameter : (BS, seq_len, 100) : (BS, 12, 100)
        '''
        parameter = self.parameter_map(raw_parameter) # (BS, 12, 256)
        batch_size = parameter.shape[0]
        # add cls_token
        if self.exp_pool == 'cls':
            cls_token = einops.repeat(self.parameter_cls_token, '() n e -> b n e', b=batch_size).to(self.device)
            parameter = torch.cat([cls_token, parameter], dim=1) # (BS,13,256)

        mask = None
        if self.pos_encoding == 'sinusoidal' :
            parameter = self.parameter_pos_encoder(parameter)
        elif self.pos_encoding == 'alibi' :
            B,T = parameter.shape[:2] # B,T
            mask = self.parameter_mask[:, :T, :T].clone() \
                .detach().to(device=parameter.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
  
        output = self.parameter_encoder(parameter, mask=mask) # (BS, 13, 256)
        
        if self.exp_pool == 'avg':
            output = torch.mean(output, dim=1)
        elif self.exp_pool == 'max':
            output = torch.max(output, dim=1).values
        elif self.exp_pool == 'gpo':
            output,_ = self.exp_aggr(output)
        elif self.exp_pool == 'cls':
            output = output[:,0,:] # (BS,512)  
        else:
            raise ValueError('Invalid expression pooling method') 

        return output