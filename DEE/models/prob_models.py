import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Config, Wav2Vec2Processor, AutoConfig
from .wav2vec import Wav2Vec2Model
from funasr import AutoModel
import torch.nn.functional as F
import copy
import einops
# pip install fairseq 
import fairseq 
from dataclasses import dataclass
from .aggr import GPO, AvgPool

import sys

from DEE.utils.masks import init_alibi_biased_mask_future
from DEE.utils.pcme import MCSoftContrastiveLoss
from DEE.utils.loss import ClosedFormSampledDistanceLoss

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
    
class DynamicPoolingLayer(nn.Module):
    def __init__(self, pool_type):
        super(DynamicPoolingLayer, self).__init__()
        self.pool_type = pool_type
    def forward(self, x):
        if self.pool_type == 'avg':
            return torch.mean(x, dim=1)
        elif self.pool_type == 'max':
            return torch.max(x, dim=1).values
        else:
            raise NotImplementedError('pool type {} is not implemented'.format(self.pool_type))
        
class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
        self.dummy = nn.Identity()
    def forward(self, x, mask=None):
        return self.dummy(x)
    
class ProbAudioEncoder(nn.Module): 
    def __init__(self, args) :
        super(ProbAudioEncoder, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.audio_max_seq_len = args.max_seq_len * 50
        self.num_audio_layers = args.num_audio_layers
        self.no_audio_PE = args.no_audio_PE # don't use positional encoding
        self.pos_encoding = args.pos_encoding
        self.use_embeddings = args.use_embeddings # extract embedding from audio
        self.input_embeddings = args.input_embeddings # use embedding as input
        self.add_n_mog = args.add_n_mog # number of gaussians -> default is 0 which means single gaussian
        self.unfreeze_emo2vec = args.unfreeze_emo2vec
        self.unfreeze_block_layer = args.unfreeze_block_layer
        if not self.input_embeddings: # if input audio
            model_dir = '../DEE/models/emo2vec'
            model_path = UserDirModule(model_dir)
            emo2vec_checkpoint = '../DEE/models/emo2vec/emotion2vec_base.pt'
            fairseq.utils.import_user_module(model_path)
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([emo2vec_checkpoint])
            self.embedding_model = model[0]
            # if we are inputting audio and using embeddings, freeze embedding_model
            if args.use_finetuned_emo2vec:
                # finetuned_model = AutoModel(model="iic/emotion2vec_base_finetuned", model_revision="v2.0.4")
                finetuned_model = AutoModel(model="iic/emotion2vec_base_finetuned")
                finetuned_model_state_dict = finetuned_model.model.state_dict()
                pt_model_state_dict = self.embedding_model.state_dict()
                new_checkpoint = {k: v for k, v in finetuned_model_state_dict.items() if k in list(pt_model_state_dict.keys())}
                self.embedding_model.load_state_dict(new_checkpoint)

            for param in self.embedding_model.parameters():
                param.requires_grad = False
            self.embedding_model.eval()
            
            if self.unfreeze_emo2vec:
                for param in self.embedding_model.parameters():
                    param.requires_grad = True # unfreeze all
                for param in self.embedding_model.modality_encoders['AUDIO'].parameters(): # freeze feature extractor
                    param.requires_grad = False
                if self.unfreeze_block_layer != 0 : # if unfreeze_block_layer is 0, then unfreeze all layers
                    print('unfreezing only the first {} transformer blocks'.format(self.unfreeze_block_layer))
                    for layer_num in range(self.unfreeze_block_layer):
                        for param in self.embedding_model.blocks[layer_num].parameters():
                            param.requires_grad = False
                self.embedding_model.train()
                
        if args.num_audio_layers == 0:
            self.audio_encoder = IdentityLayer()
        else:
            self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=args.num_audio_heads, batch_first=True)
            self.audio_encoder = nn.TransformerEncoder(self.audio_encoder_layer, num_layers=args.num_audio_layers)
        
        self.audio_mean_layer = nn.TransformerEncoderLayer(d_model=768, nhead=args.num_audio_heads, batch_first=True)
        self.audio_mean_encoder = nn.TransformerEncoder(self.audio_mean_layer, num_layers=1)
        # self.audio_mean_encoder = nn.Identity() # the last layer of the embedding model is the mean encoder
        
        self.audio_uncertainty_layer = nn.TransformerEncoderLayer(d_model=768, nhead=args.num_audio_heads, batch_first=True)
        self.audio_uncertainty_encoder = nn.TransformerEncoder(self.audio_uncertainty_layer, num_layers=1)
        
        self.std_feature_map = nn.Linear(768, args.feature_dim)
        self.mean_feature_map = nn.Linear(768, args.feature_dim)
        
        if self.add_n_mog >= 1:
            self.mog_means_layers = nn.ModuleList([nn.Linear(768, args.feature_dim) for _ in range(self.add_n_mog)])
            self.mog_logvars_layers = nn.ModuleList([nn.Linear(768, args.feature_dim) for _ in range(self.add_n_mog)])
        
        # should the pooling operation be the same when we use mog?
        self.audio_pool = args.audio_pool
        if self.audio_pool == 'gpo':
            self.mean_pool = GPO(32, 32)
        elif self.audio_pool == 'avg' or self.audio_pool == 'max':
            self.mean_pool = DynamicPoolingLayer(self.audio_pool)
            
        self.std_pool = GPO(32, 32) # use GPO for std head as we are not sure what to use
        
        if not self.no_audio_PE:
            if self.pos_encoding == 'sinusoidal' :
                self.audio_pos_encoder = PositionalEncoding(768, self.audio_max_seq_len)
            elif self.pos_encoding == 'alibi' :
                self.audio_mask = init_alibi_biased_mask_future(args.num_audio_heads, self.audio_max_seq_len)
        
    def forward_base(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        if not self.input_embeddings: # if input audio
            if not self.unfreeze_emo2vec:
                self.embedding_model.eval() # always set to eval mode
            embeddings = self.embedding_model.extract_features(audio, padding_mask=None) 
            embeddings = embeddings['x'] # (BS, 50*sec, 768)
        else:
            embeddings = audio
        feats = embeddings
        feats = feats[:, :self.audio_max_seq_len, :] # (BS, 50*sec, 768)
        mask = None
        if (not self.no_audio_PE) and (not self.num_audio_layers == 0):
            if self.pos_encoding == 'sinusoidal' :
                feats = self.audio_pos_encoder(feats)# [BS, seq_len, 768]
            elif self.pos_encoding == 'alibi' :
                B,T = feats.shape[:2] # B,T
                mask = self.audio_mask[:, :T, :T].clone() \
                    .detach().to(device=feats.device)
                if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                    mask = mask.repeat(B, 1, 1)
                    
        feats = self.audio_encoder(feats, mask=mask) # (BS, 50*sec, 768)
        std_feats = feats
        mean_feats = feats
        return std_feats, mean_feats

    def forward_mean(self, features):
        encoder_features = self.audio_mean_encoder(features) # one layer transformer encoder
        
        if self.audio_pool != 'gpo': # according to PCME++, gpo is done after the feature map
            encoder_features = self.mean_pool(encoder_features) # max or avg pool
                
        features = self.mean_feature_map(encoder_features)
        
        if self.audio_pool == 'gpo':
            features, _ = self.mean_pool(features)
        
        if self.add_n_mog >= 1: #(BS, seq_len, feature_dim) -> (BS, feature_dim) -> (BS, add_n_mog, feature_dim)
            if self.audio_pool != 'gpo':
                mog_means = torch.stack([layer(encoder_features) for layer in self.mog_means_layers], dim=1)
            elif self.audio_pool == 'gpo':
                mog_means = torch.stack([self.mean_pool(layer(encoder_features))[0] for layer in self.mog_means_layers], dim=1)
            features = torch.cat([features.unsqueeze(1), mog_means], dim=1)
            # (BS, add_n_mog+1, feature_dim)
        
        mean = l2norm(features, dim=-1)
        return mean 
    
    def forward_std(self, features):
        encoder_features = self.audio_uncertainty_encoder(features)
        features = self.std_feature_map(encoder_features)
        pooled_features, _  = self.std_pool(features)# BS, feature_dim
        if self.add_n_mog >= 1: # layer(features) -> (BS, feature_dim) -> stack -> (BS, add_n_mog, feature_dim)
            mog_logvars  = torch.stack([self.std_pool(layer(encoder_features))[0] for layer in self.mog_logvars_layers], dim=1)
            pooled_features = torch.cat([pooled_features.unsqueeze(1), mog_logvars], dim=1)
            # (BS, add_n_mog+1, feature_dim)
            
        logvar = pooled_features # std has no l2norm
        return logvar
    
    def forward(self, audio):
        std_feats, mean_feats = self.forward_base(audio)
        logvar = self.forward_std(std_feats)
        mean = self.forward_mean(mean_feats)
        # self.add_n_mogs >=1 leads to (BS, add_n_mog+1, feature_dim)
        # self.add_n_mogs ==0 leads to (BS, feature_dim)
        return mean, logvar 
    
class ProbExpEncoder(nn.Module): 
    def __init__(self, args) :
        super(ProbExpEncoder, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.fps = 30 if args.use_30fps else 25
        self.exp_max_seq_len = args.max_seq_len * self.fps
        self.pos_encoding = args.pos_encoding
        self.add_n_mog = args.add_n_mog # number of gaussians -> default is 0 which means single gaussian
        ## For parameter encoder
        self.exp_map = nn.Linear(args.parameter_dim, args.parameter_feature_dim)
        if self.pos_encoding == 'sinusoidal' :
            self.exp_pos_encoder = PositionalEncoding(args.parameter_feature_dim, self.exp_max_seq_len)
        elif self.pos_encoding == 'alibi' :
            self.exp_mask = init_alibi_biased_mask_future(args.num_parameter_heads, self.exp_max_seq_len)
            
        self.exp_encoder_layer = nn.TransformerEncoderLayer(d_model=args.parameter_feature_dim, nhead=args.num_parameter_heads, batch_first=True)
        self.exp_encoder = nn.TransformerEncoder(self.exp_encoder_layer, num_layers=args.num_parameter_layers)

        self.exp_mean_layer = nn.TransformerEncoderLayer(d_model=args.parameter_feature_dim, nhead=args.num_parameter_heads, batch_first=True)
        self.exp_mean_encoder = nn.TransformerEncoder(self.exp_mean_layer, num_layers=1)
        
        self.exp_uncertainty_layer = nn.TransformerEncoderLayer(d_model=args.parameter_feature_dim, nhead=args.num_parameter_heads, batch_first=True)
        self.exp_uncertainty_encoder = nn.TransformerEncoder(self.exp_uncertainty_layer, num_layers=1)
        
        self.std_feature_map = nn.Linear(args.parameter_feature_dim, args.feature_dim)
        self.mean_feature_map = nn.Linear(args.parameter_feature_dim, args.feature_dim)
        
        if self.add_n_mog >= 1:
            self.mog_means_layers = nn.ModuleList([nn.Linear(args.parameter_feature_dim, args.feature_dim) for _ in range(self.add_n_mog)])
            self.mog_logvars_layers = nn.ModuleList([nn.Linear(args.parameter_feature_dim, args.feature_dim) for _ in range(self.add_n_mog)])
            
        self.exp_pool = args.exp_pool
        if self.exp_pool == 'gpo':
            self.mean_pool = GPO(32, 32)
        elif self.exp_pool == 'avg' or self.exp_pool == 'max':
            self.mean_pool = DynamicPoolingLayer(self.exp_pool)
        self.std_pool = GPO(32, 32) # use GPO for std head as we are not sure what to use
        
        
    def forward_base(self, raw_exp) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        exp = self.exp_map(raw_exp) # (BS, seq_len, model_dim)
        mask = None
        if self.pos_encoding == 'sinusoidal' :
            exp = self.exp_pos_encoder(exp)# [BS, seq_len, 768]
        elif self.pos_encoding == 'alibi' :
            B,T = exp.shape[:2] # B,T
            mask = self.exp_mask[:, :T, :T].clone() \
                .detach().to(device=exp.device)
            if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                mask = mask.repeat(B, 1, 1)
        # output = self.audio_encoder(exp, mask=mask) # [BS, seq_len, 768]
        feats = self.exp_encoder(exp, mask=mask) 
        std_feats = feats
        mean_feats = feats
        return std_feats, mean_feats

    def forward_mean(self, features):
        encoder_features = self.exp_mean_encoder(features) # [BS, seq_len, dim]
        
        if self.exp_pool != 'gpo': # according to PCME++, gpo is done after the feature map
            encoder_features = self.mean_pool(encoder_features) # max or avg pool
        
        features = self.mean_feature_map(encoder_features) # [BS, seq_len, feat_dim]
        
        if self.exp_pool == 'gpo':
            features, _ = self.mean_pool(features) # [BS, feat_dim]
        
        if self.add_n_mog >= 1: #(BS, seq_len, feature_dim) -> (BS, feature_dim) -> (BS, add_n_mog, feature_dim)
            if self.exp_pool != 'gpo':
                mog_means = torch.stack([layer(encoder_features) for layer in self.mog_means_layers], dim=1)
            elif self.exp_pool == 'gpo':
                mog_means = torch.stack([self.mean_pool(layer(encoder_features))[0] for layer in self.mog_means_layers], dim=1)
            features = torch.cat([features.unsqueeze(1), mog_means], dim=1)
            
        mean = l2norm(features, dim=-1)
        return mean 
    
    def forward_std(self, features):
        encoder_features = self.exp_uncertainty_encoder(features)
        features = self.std_feature_map(encoder_features)
        pooled_features, _  = self.std_pool(features)
        
        if self.add_n_mog >= 1: # layer(features) -> (BS, feature_dim)
            mog_logvars = torch.stack([self.std_pool(layer(encoder_features))[0] for layer in self.mog_logvars_layers], dim=1)
            pooled_features = torch.cat([pooled_features.unsqueeze(1), mog_logvars], dim=1)
            # (BS, add_n_mog+1, feature_dim)
              
        logvar = pooled_features
        return logvar
    
    def forward(self, audio):
        std_feats, mean_feats = self.forward_base(audio)
        logvar = self.forward_std(std_feats)
        mean = self.forward_mean(mean_feats)
        # self.add_n_mogs >=1 leads to (BS, add_n_mog+1, feature_dim)
        # self.add_n_mogs ==0 leads to (BS, feature_dim)
        return mean, logvar
    
class ProbDEE(nn.Module):
    def __init__(self, args):
        super(ProbDEE, self).__init__()
        self.audio_encoder = ProbAudioEncoder(args)
        self.exp_encoder = ProbExpEncoder(args)
        if args.loss == 'csd':
            if args.add_n_mog >= 1:
                raise Exception('CSD loss does not support MOG')
            self.criterion = ClosedFormSampledDistanceLoss(
                            init_shift=5, 
                            init_negative_scale=5,
                            vib_beta=args.vib_beta,
                            smoothness_alpha=args.smoothness_alpha)
        elif args.loss == 'soft_contrastive':
            self.criterion = MCSoftContrastiveLoss(
                            args,
                            init_shift=5, 
                            init_negative_scale=5,
                            num_samples = args.num_samples,
                            vib_beta=args.vib_beta)
        else:
            raise NotImplementedError('loss {} is not implemented'.format(args.loss))
        
    def forward(self, audio, raw_exp):
        audio_mean, audio_logvar = self.audio_encoder(audio)
        audio_emb = {}
        audio_emb['mean'] = audio_mean
        audio_emb['std'] = audio_logvar # actually log var but the name is kept for consistency
        
        exp_mean, exp_logvar = self.exp_encoder(raw_exp)
        exp_emb = {}
        exp_emb['mean'] = exp_mean
        exp_emb['std'] = exp_logvar # actually log var but the name is kept for consistency
        
        return audio_emb, exp_emb
    
