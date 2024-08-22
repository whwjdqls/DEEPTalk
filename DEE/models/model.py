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
from funasr import AutoModel
import sys
from .aggr import GPO, AvgPool

from DEE.utils.masks import init_alibi_biased_mask_future

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

class DEE(nn.Module) :
    def __init__(self, args) :
        super(DEE, self).__init__()
        raise NotImplementedError('This version of DEE is not supported. Please use DEE_v2 instead.')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.no_cls = args.no_cls
        self.audio_max_seq_len = args.max_seq_len * 50 # audio feature is 50HZ
        self.exp_max_seq_len = args.max_seq_len * 30 # expression param is 30HZ
        self.use_embeddings = args.use_embeddings # use wav2vec2 embeddings only
        self.pos_encoding = args.pos_encoding # 'alibi' or 'sinusoidal'
        self.audio_mask = None
        self.parameter_mask = None

        # compute loss
        if args.temperature_fix :
            self.logit_scale = torch.ones([]) * np.log(1/args.temperature)
        else :
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/args.temperature))

        ## For audio encoder
        # Change number of hidden layers for wav2vec2
        self.num_audio_layers = args.num_audio_layers
        if args.use_SER_encoder : # use SER encoder for audio encoder 
            config = AutoConfig.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition", num_hidden_layers = args.num_audio_layers)
            self.audio_encoder = Wav2Vec2Model.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition", config=config)
            self.audio_feature_map = nn.Linear(1024, args.feature_dim, bias=False)
            
        elif args.use_embeddings:
            # frozen wav2vec2 model for extracting embeddings
            embedding_model_config = Wav2Vec2Config(num_hidden_layers=12) # use all layers
            self.embedding_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h",config=embedding_model_config)
            for param in self.embedding_model.parameters():
                param.requires_grad = False
            self.embedding_model.eval()
            self.weight = nn.Parameter(torch.ones(13, 1))
            
            if self.pos_encoding == 'sinusoidal' :
                self.audio_pos_encoder = PositionalEncoding(768, self.audio_max_seq_len)
            elif self.pos_encoding == 'alibi' :
                self.audio_mask = init_alibi_biased_mask_future(args.num_audio_heads, self.audio_max_seq_len)
            
 
            self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=args.num_audio_heads, batch_first=True)
            self.audio_encoder = nn.TransformerEncoder(self.audio_encoder_layer, num_layers=args.num_audio_layers)
            self.audio_feature_map = nn.Linear(768, args.feature_dim, bias=False)
        else : # use wav2vec2
            config = Wav2Vec2Config(num_hidden_layers=self.num_audio_layers)
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", config=config)
            self.audio_feature_map = nn.Linear(768, args.feature_dim, bias=False)
            
        # if self.no_cls :
        #     embedding_len = round(49*args.audio_feature_len/16000)+1
        #     self.audio_temp_map = nn.Linear(embedding_len,1)
        
        if args.freeze_FE :
            self.audio_encoder.feature_extractor._freeze_parameters()


        ## For parameter encoder
        self.parameter_map = nn.Linear(args.parameter_dim, args.parameter_feature_dim)
        if self.pos_encoding == 'sinusoidal' :
            self.parameter_pos_encoder = PositionalEncoding(args.parameter_feature_dim, self.exp_max_seq_len)
        elif self.pos_encoding == 'alibi' :
            self.parameter_mask = init_alibi_biased_mask_future(args.num_parameter_heads, self.exp_max_seq_len)
        self.cls_token = nn.Parameter(torch.rand(1,1,args.parameter_feature_dim))
        self.parameter_encoder_layer = nn.TransformerEncoderLayer(d_model=args.parameter_feature_dim, nhead=args.num_parameter_heads, batch_first=True)
        self.parameter_encoder = nn.TransformerEncoder(self.parameter_encoder_layer, num_layers=args.num_parameter_layers)
        self.parameter_feature_map = nn.Linear(args.parameter_feature_dim, args.feature_dim, bias=False)

        # self.apply(init_normal)
        if args.param_initialize :
            self.initialize_parameters(args.use_speaker_norm)

    def initialize_parameters(self, use_speaker_norm):
        nn.init.normal_(self.audio_feature_map.weight, std=0.02)
        nn.init.normal_(self.parameter_map.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.01)
        nn.init.normal_(self.parameter_feature_map.weight, std=0.02)

        for param in self.parameter_encoder.parameters():
            if len(param.shape) > 1:
                nn.init.normal_(param, mean=0, std=0.02)
        if use_speaker_norm :
            nn.init.normal_(self.SID_layer.weight, std=0.02)

    def encode_audio(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        
        if self.use_embeddings:
            # embedding model should not use class tokens
            hidden_states = self.embedding_model(audio,output_hidden_states=True, no_cls=1)['hidden_states'] # 7 hidden states (tuple)
            hidden_states = torch.stack(hidden_states) # [13, BS,seq_len, 768]
            hidden_states = hidden_states.permute(1,2,3,0) # [BS, seq_len, 768, 13]
            output = hidden_states @ self.weight # [BS, seq_len, 768, 1]
            output = output / self.weight.sum() # [BS, seq_len, 768, 1]
            output = output.squeeze(3) # [BS, seq_len, 768]
            mask = None
            if self.pos_encoding == 'sinusoidal' :
                output = self.audio_pos_encoder(output)# [BS, seq_len, 768]
            elif self.pos_encoding == 'alibi' :
                B,T = output.shape[:2] # B,T
                mask = self.audio_mask[:, :T, :T].clone() \
                    .detach().to(device=output.device)
                if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                    mask = mask.repeat(B, 1, 1)
            output = self.audio_encoder(output, mask=mask) # [BS, seq_len, 768]
        else: # 
            output = self.audio_encoder(audio, no_cls=self.no_cls).last_hidden_state # (BS,50*sec,768) : (BS,20,768)
        output = self.audio_feature_map(output) # (BS,50*sec,768) : (BS,20,512)
        
        # take features from the cls embedding
        if not self.no_cls : # -> use cls token
            output = output[:,0,:] # (BS,512)
        else : # -> avg pooling over time
            output = torch.mean(output, dim=1) # (BS, 512)

        return output


    def encode_parameter(self, raw_parameter) :
        '''
        raw_parameter : (BS, seq_len, 100) : (BS, 12, 100)
        '''
        parameter = self.parameter_map(raw_parameter) # (BS, 12, 256)
        batch_size = parameter.shape[0]
        # add cls_token
        # cls_token = nn.Parameter(torch.rand(1,1,256))
        cls_token = einops.repeat(self.cls_token, '() n e -> b n e', b=batch_size).to(self.device)
        parameter = torch.cat([cls_token, parameter], dim=1) # (BS,13,256)
        # add positional encoding
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
        output = self.parameter_feature_map(output) # (BS, 13, 512)
        output = output[:,0,:] # (BS,512)
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
    
class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
        self.dummy = nn.Identity()
    def forward(self, x, mask=None):
        return self.dummy(x)
        
# model for classification
class SIDetector(nn.Module):
    def __init__(self, args) :
        super(SIDetector, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.gender_norm :
            self.speaker_num = 2
        else :
            if args.use_RAVDESS :
                self.speaker_num = 24
            elif args.use_MEAD :
                self.speaker_num = 32
                
        self.fc1 = nn.Linear(args.feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.speaker_num)
        
    def forward(self, audio_embedding) :
        output = torch.relu(self.fc1(audio_embedding))
        output = torch.relu(self.fc2(output))
        output = self.fc3(output)
        return output
# model for classification
class SIDetector_grl(nn.Module):
    def __init__(self, args) :
        super(SIDetector_grl, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grl = GradientReversal(alpha=args.SID_lambda)
        if args.gender_norm :
            self.speaker_num = 2
        else :
            if args.use_RAVDESS :
                self.speaker_num = 24
            elif args.use_MEAD :
                self.speaker_num = 32
                
        self.fc1 = nn.Linear(args.feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.speaker_num)
        
    def forward(self, audio_embedding) :
        audio_embedding = self.grl(audio_embedding)
        output = torch.relu(self.fc1(audio_embedding))
        output = torch.relu(self.fc2(output))
        output = self.fc3(output)
        return output  
@dataclass
class UserDirModule:
    user_dir: str
    
# this is a version of DEE that uses emotion2vec as the pretrained audio_encoder
class DEE_v2(nn.Module) :
    def __init__(self, args, load_checkpoint=True) :
        super(DEE_v2, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.use_embeddings = args.use_embeddings
        self.no_audio_PE = args.no_audio_PE
        self.audio_pool = args.audio_pool
        self.exp_pool = args.exp_pool
        # self.no_cls = args.no_cls # no_cls means not using cls token for audio embedding, but use cls token for parameter embedding
        # # if no_cls is false, then cls token is used for audio embedding and the parameter embedding
        # self.max_pool = args.max_pool# to make compatiblw with the previous version, max_pooling maxpools audio and expression features
        # self.gpo = args.gpo # to make compatible with the previous version, gpo is uses gpo for audio and expression features
        self.audio_max_seq_len = args.max_seq_len * 50 # audio feature is 50HZ
        self.exp_max_seq_len = args.max_seq_len * 30 # expression param is 30HZ
        self.pos_encoding = args.pos_encoding # 'alibi' or 'sinusoidal' 
        self.audio_mask = None
        self.parameter_mask = None
        self.unfreeze_emo2vec = args.unfreeze_emo2vec
        self.unfreeze_block_layer = args.unfreeze_block_layer
        # compute loss
        if args.temperature_fix :
            self.logit_scale = torch.ones([]) * np.log(1/args.temperature)
        else :
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/args.temperature))
        model_dir = '../DEE/models/emo2vec'
        model_path = UserDirModule(model_dir)
        emo2vec_checkpoint = '../DEE/models/emo2vec/emotion2vec_base.pt'
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([emo2vec_checkpoint])
        self.embedding_model = model[0]
        if args.use_finetuned_emo2vec:
            finetuned_model = AutoModel(model="iic/emotion2vec_base_finetuned", model_revision="v2.0.4")
            finetuned_model_state_dict = finetuned_model.model.state_dict()
            pt_model_state_dict = self.embedding_model.state_dict()
            new_checkpoint = {k: v for k, v in finetuned_model_state_dict.items() if k in list(pt_model_state_dict.keys())}
            self.embedding_model.load_state_dict(new_checkpoint)
        
        for param in self.embedding_model.parameters():
            param.requires_grad = False # first freeze the model
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
        
        # embedding_len = round(49*args.audio_feature_len/16000)+1
        if not self.no_audio_PE:
            if self.pos_encoding == 'sinusoidal' :
                self.audio_pos_encoder = PositionalEncoding(768, self.audio_max_seq_len)
            elif self.pos_encoding == 'alibi' :
                self.audio_mask = init_alibi_biased_mask_future(args.num_audio_heads, self.audio_max_seq_len)
        self.audio_cls_token = nn.Parameter(torch.rand(1,1,768)) # just initialize it for the sake of compatibility
        
        if args.num_audio_layers == 0 : 
            self.audio_encoder = IdentityLayer()
        else :            
            self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=args.num_audio_heads, batch_first=True)
            self.audio_encoder = nn.TransformerEncoder(self.audio_encoder_layer, num_layers=args.num_audio_layers)
        self.audio_feature_map = nn.Linear(768, args.feature_dim, bias=False)

        # self.audio_temp_map = nn.Linear(embedding_len,1)

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
        # self.apply(init_normal)
        if args.param_initialize :
            self.initialize_parameters(args.use_speaker_norm)
        
        

    def initialize_parameters(self, use_speaker_norm):
        nn.init.normal_(self.audio_feature_map.weight, std=0.02)
        nn.init.normal_(self.parameter_map.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.01)
        nn.init.normal_(self.parameter_feature_map.weight, std=0.02)

        for param in self.parameter_encoder.parameters():
            if len(param.shape) > 1:
                nn.init.normal_(param, mean=0, std=0.02)
        if use_speaker_norm :
            nn.init.normal_(self.SID_layer.weight, std=0.02)

    def encode_audio(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 6400)
        '''
        if not self.unfreeze_emo2vec:
            self.embedding_model.eval() # always set to eval mode
        feats = self.embedding_model.extract_features(audio, padding_mask=None) 
        feats = feats['x'] # (BS, 50*sec, 768)
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

