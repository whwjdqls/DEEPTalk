import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Config, Wav2Vec2Processor, AutoConfig
from .wav2vec import Wav2Vec2Encoder
import torch.nn.functional as F
from .TVAE_inferno import TVAE
from .temporal.TransformerMasking import init_faceformer_biased_mask_future
from .base_models import Norm, Residual, Attention
import os, sys

sys.path.append(f'../')
from DEE.utils.pcme import sample_gaussian_tensors
def sample_gaussian_tensors_test(mu, logsigma, num_samples, normalize=False, control_logvar=None):
    if num_samples == 0:
        return mu.unsqueeze(1)
    if control_logvar:
        logsigma = torch.ones_like(logsigma) * (control_logvar)
    logsigma = logsigma * 0.5
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), # BS, num_samples, dim
                      dtype=mu.dtype, device=mu.device)

    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
        mu.unsqueeze(1))
    if normalize:
        return F.normalize(samples, p=2, dim=-1)
    else:
        return samples


# import copy
# import einops
# from .sequence_encoder import LinearSequenceEncoder, ConvSquasher, StackLinearSquash
# from inferno.models.talkinghead.FaceFormerDecoder import BertPriorDecoder 
# from omegaconf import open_dict

# from VAEs import TVAE
def calculate_vertice_loss(pred, target):
     reconstruction_loss = nn.MSELoss()(pred, target)
     return reconstruction_loss
 
                    ##(linear stack, 128 *2,  128,   3,          4)
def _create_squasher(type, input_dim, output_dim, quant_factor, latent_frame_size =4): 
    if type == "conv": 
        return ConvSquasher(input_dim, quant_factor, output_dim)
    elif type == "stack_linear": 
        return StackLinearSquash(input_dim, latent_frame_size, output_dim)
    else: 
        raise ValueError("Unknown squasher type")

# class SALN_Transformer(torch.nn.Modeule) :
#     def __init__(self, scale, bias, in_size=50, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
#                 intermediate_size=3072, in_dim2=None) :
#         super().__init__()
#         blocks = []
#         attn = False

#         layer = [Residual(scale*Norm(Attention(in_size, hidden_size,
#                                             heads=num_attention_heads), hidden_size)+bias)
#                 Residual(scale*Norm)]   

#         for i in range(num_hidden_layers) :
#             blocks.extend([])

class SequenceEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        
    def forward(self, sample):
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError("Subclasses must implement this method")

    def input_feature_dim(self):
        raise NotImplementedError("Subclasses must implement this method")

    def output_feature_dim(self):
        raise NotImplementedError("Subclasses must implement this method")

class LinearSequenceEncoder(SequenceEncoder): 

    def __init__(self, input_feature_dim, output_feature_dim):
        super().__init__()
        # self.cfg = cfg
        # input_feature_dim = self.cfg.get('input_feature_dim', None) or self.cfg.feature_dim 
        # output_feature_dim = self.cfg.feature_dim
        self.linear = torch.nn.Linear(input_feature_dim, output_feature_dim)

    def forward(self, sample):
        # feat = sample[input_key] 
        # B, T, D -> B * T, D 
        feat = sample.view(-1, sample.shape[-1]) # (BS*64,768)
        out = self.linear(feat) # (BS*64,128)
        # B * T, D -> B, T, D
        out = out.view(sample.shape[0], sample.shape[1], -1) # (BS,64,128)
        return out

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim

class ConvSquasher(nn.Module): 

    def __init__(self, input_dim, quant_factor, output_dim) -> None:
        super().__init__()
        self.squasher = create_squasher(input_dim, output_dim, quant_factor)

    def forward(self, x):
        # BTF -> BFT 
        x = x.transpose(1, 2)
        x = self.squasher(x)
        # BFT -> BTF
        x = x.transpose(1, 2)
        return x

class StackLinearSquash(nn.Module): #( 128 *2, 4, 128)
    def __init__(self, input_dim, latent_frame_size, output_dim): 
        super().__init__()
        self.input_dim = input_dim # 128*2 
        self.latent_frame_size = latent_frame_size  # 4
        self.output_dim = output_dim # 128
        self.linear = nn.Linear(input_dim * latent_frame_size, output_dim)
        # print(f'input dim : {self.input_dim * latent_frame_size}')
        
    def forward(self, x):
        B, T, F = x.shape # (BS,64,256)
        # input B, T, F -> B, T // latent_frame_size, F * latent_frame_size
        assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
        T_latent = T // self.latent_frame_size
        F_stack = F * self.latent_frame_size
        x = x.reshape(B, T_latent, F_stack) # (BS,16,1024)
        x = x.view(B * T_latent, F_stack)
        x = self.linear(x)
        x = x.view(B, T_latent, -1)
        return x

class LinearEmotionCondition(nn.Module):
    def __init__(self, condition_dim, output_dim):
        super().__init__()
        self.map = nn.Linear(condition_dim, output_dim)

    def forward(self, sample):
        return self.map(sample)



class DEMOTE(nn.Module) :
    def __init__(self, EMOTE_config, FLINT_config, DEE_config, FLINT_ckpt, DEE, load_motion_prior=True) :
        super(DEMOTE, self).__init__()
        ## audio encoder
        self.audio_model = Wav2Vec2Encoder(EMOTE_config['audio_config']['model_specifier'], 
            EMOTE_config['audio_config']['trainable'], 
            with_processor=EMOTE_config['audio_config']['with_processor'], 
            expected_fps=EMOTE_config['audio_config']['model_expected_fps'], # 50 fps is the default for wav2vec2 (but not sure if this holds universally)
            target_fps=EMOTE_config['audio_config']['target_fps'], # 25 fps is the default since we use 25 fps for the videos 
            freeze_feature_extractor=EMOTE_config['audio_config']['freeze_feature_extractor'])
        input_feature = self.audio_model.output_feature_dim() #768
        # sequence encoder
        decoder_config = EMOTE_config['sequence_decoder_config']
        self.sequence_encoder = LinearSequenceEncoder(input_feature, decoder_config['feature_dim'])
        self.sequence_decoder = BertPriorDecoder(decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior)
        self.DEE_config = DEE_config
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
    def encode_audio(self, audio) :
        '''
        audio : (BS, seq_length) : (BS, 40960)
        '''
        audio = self.processor(audio,sampling_rate=16000, return_tensors="pt").input_values[0].to(audio.device)
        output = self.audio_model(audio) # (BS,64,768)
        output = self.sequence_encoder(output) # (BS,64,128)
        return output # (BS,64,128)

    def forward(self, audio_content, condition, audio_emotion=None, sample=False, control_logvar=None) :
        '''
        (NOTE) audio_content and audio_emotion are unprocessed raw audio samples 
        audio_content : (BS, seq_length) : (BS, 40960) 
        condition : (BS, condition_num)
        '''
        audio_embedding = self.encode_audio(audio_content) # (BS,64,128) # this is wav2vec2 conv features

        if audio_emotion == None :# audio_content and audio_emotion are the same
            audio_emotion = audio_content
        else : 
            print('Other audio for conditioning emotion used')
        
        if self.DEE_config.process_type == 'layer_norm': #
            audio_emotion = torch.nn.functional.layer_norm(audio_emotion,(audio_emotion[0].shape[-1],))
        elif self.DEE_config.process_type == 'wav2vec2':
            audio_emotion = self.processor(audio_emotion,sampling_rate=16000, return_tensors="pt").input_values[0]
        else:
            raise ValueError('DEE_config.process_type should be layer_norm or wav2vec2')
        
        output = self.sequence_decoder(condition, audio_emotion, audio_embedding, sample=sample, control_logvar=control_logvar) # (BS,128,53)
        return output # (BS,128,53)


class BertPriorDecoder(nn.Module):
    def __init__(self, decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior):
        super(BertPriorDecoder, self).__init__()

        ## style encoder
        style_config = decoder_config['style_embedding']
        style_dim = style_config['n_intensities'] + style_config['n_identities'] + style_config['n_expression'] # 43
        identity_dim = int(style_config['n_identities'])
        
        # print(f'style dim : {style_dim}')
        self.obj_vector = LinearEmotionCondition(style_dim, decoder_config['feature_dim'])
        ## decoder
        #mask
        max_len = 1200
        self.biased_mask = init_faceformer_biased_mask_future(num_heads = decoder_config['nhead'], max_seq_len = max_len, period=decoder_config['period'])
        # transformer encoder
        try :
            self.add_condition = decoder_config['add_condition']
            print(f'add condition : {self.add_condition}')
        except :
            self.add_condition = False
        if self.add_condition :
            dim_factor = 1
        else :
            dim_factor = 2
        print(f'dim factor : {dim_factor}')
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=decoder_config['feature_dim'] * dim_factor, 
                    nhead=decoder_config['nhead'], dim_feedforward=dim_factor*decoder_config['feature_dim'], 
                    activation=decoder_config['activation'],
                    dropout=decoder_config['dropout'], batch_first=True
        )        
        self.bert_decoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=decoder_config['num_layers'])
        # decoder.decoder
        self.decoder = nn.Linear(dim_factor*decoder_config['feature_dim'], decoder_config['feature_dim'])
        # Squasher
        if decoder_config['squash_after'] : #(linear stack, 128 *2, 128,3, 4)
            self.squasher_2 = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], decoder_config['feature_dim'], decoder_config['quant_factor'], decoder_config['latent_frame_size'])
        elif decoder_config['squash_before'] :
            self.squasher_1 = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'], decoder_config['latent_frame_size'])
        else : 
            raise ValueError("Unknown squasher type")

        

        # Temporal VAE decoder
        # 11-21
        # Load only decoder from TVAE
        self.motion_prior = TVAE(FLINT_config).motion_decoder
        # Assuming self.motion_prior is a PyTorch model

        if load_motion_prior :
            print(f'Load FLINT checkpoints from {FLINT_ckpt}')
            decoder_ckpt = torch.load(FLINT_ckpt)
            if 'state_dict' in decoder_ckpt:
                decoder_ckpt = decoder_ckpt['state_dict']
            # new_decoder_ckpt = decoder_ckpt.copy()
            motion_decoder_state_dict = {
                key.replace('motion_decoder.', ''): value
                for key, value in decoder_ckpt.items()
                if key.startswith('motion_decoder.')
            }
            self.motion_prior.load_state_dict(motion_decoder_state_dict)
        
        # freeze decoder
        self.motion_prior.eval()
        for param in self.motion_prior.parameters():
            param.requires_grad = False
        
        # Implement DEE
        self.DEE = DEE
        if self.DEE.__class__.__name__ == 'ProbDEE':
            self.prob_DEE = True
            self.point_DEE = False
        else:
            self.prob_DEE = False
            self.point_DEE = True
        if self.point_DEE and self.prob_DEE :
            print('choose either point_DEE or prob_DEE')
        for param in self.DEE.parameters() :
            param.requires_grad = False
        if self.point_DEE :
            self.DEE_audio = self.DEE.encode_audio
        if self.prob_DEE :
            self.num_samples = int(decoder_config["DEE"]["num_samples"])
            self.DEE_audio = self.DEE.audio_encoder
        self.emotion_map_layer = nn.Linear(128,128)
        input_dim = identity_dim + 128
        self.condition_feature_layer = nn.Linear(input_dim,128)
        # self.SALN_scale = nn.Linear(128,1)
        # self.SALN_bias = nn.Linear(128,1)



    def encode_style(self, condition) :
        '''
        condition : (BS, condition_num)
        condition_num = emotion + intensity + actors = 43
        '''
        output = self.obj_vector(condition) # (BS, condition_num)
        output = output.unsqueeze(1) # (BS,1,128)
        return output

    def encode_emotion_condition(self, audio, sample=False, control_logvar=None) : 
        
        if self.point_DEE :
            if sample:
                raise ValueError('DEE is point_DEE, so sample should be False')
            audio_emotion_embedding = self.DEE_audio(audio) # (BS,128)
            audio_emotion_embedding = audio_emotion_embedding / audio_emotion_embedding.norm(dim=1, keepdim=True)
        
        if self.prob_DEE :
            mu, logvar = self.DEE_audio(audio) # mu:(BS,128) / logvar:(BS,128)
            if not sample :
                audio_emotion_embedding = mu
            elif sample :
                # audio_emotion_embedding = sample_gaussian_tensors(mu, logvar, self.num_samples, normalize=True).squeeze(1)
                audio_emotion_embedding = sample_gaussian_tensors_test(mu, logvar, self.num_samples, normalize=True, control_logvar=control_logvar).squeeze(1)

        audio_emotion_embedding = self.emotion_map_layer(audio_emotion_embedding)
        return audio_emotion_embedding


    def decode(self, sample) :
        
        mask = self.biased_mask[:, :sample.shape[1], :sample.shape[1]].clone().detach().to(device=sample.device)
        if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
            mask = mask.repeat(sample.shape[0], 1, 1)
        
        output = self.bert_decoder(sample, mask=mask) # (BS,64,256)
        output = self.decoder(output) # (BS,64,128)
        output = self.squasher_2(output) # (BS,16,128)
        output = self.motion_prior.forward(output) 
        return output

    def forward(self, condition, audio_emotion, audio_embedding, sample=False, control_logvar=None) :

        repeat_num = audio_embedding.shape[1]
        audio_emotion_embedding = self.encode_emotion_condition(audio_emotion, sample=sample, control_logvar=control_logvar) # (BS,128)
        actor_condition_onehot = condition[:,11:] # (BS,32)
        emotion_actor_cat = torch.cat([audio_emotion_embedding, actor_condition_onehot], dim=1) # (BS,160)
        style_embedding = self.condition_feature_layer(emotion_actor_cat).unsqueeze(1).repeat(1,repeat_num,1) # (BS,T,128)

        if self.add_condition :# add
            styled_audio_add = audio_embedding + style_embedding
            output = self.decode(styled_audio_add)
        else :# cat
            styled_audio_cat = torch.cat([audio_embedding, style_embedding], dim=-1) # (BS,64,256)
            output = self.decode(styled_audio_cat) # (BS,128,53)

        return output # (BS,128,53)