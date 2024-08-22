import torch
import torch.nn as nn
import numpy as np
import math
from transformers import Wav2Vec2Config, Wav2Vec2Processor, AutoConfig
from .wav2vec import Wav2Vec2Encoder
import torch.nn.functional as F
from .TVAE_inferno import TVAE
from .VAEs import VQVAE2, VQVAE
from .temporal.TransformerMasking import init_faceformer_biased_mask_future, init_alibi_biased_mask_future
from .base_models import Norm, Residual, Attention
import os, sys

sys.path.append('../')
from DEE.utils.pcme import sample_gaussian_tensors
def sample_gaussian_tensors_test(mu, logsigma, num_samples, normalize=False, control_logvar=None):
    if num_samples == 0:
        return mu.unsqueeze(1)
    if control_logvar:
        logsigma = logsigma * (control_logvar)
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
def _create_squasher(type, input_dim, output_dim, quant_factor, latent_frame_size=4, hidden_sizes=[512, 512]): 
    if type == "conv": 
        return ConvSquasher(input_dim, quant_factor, output_dim)
    elif type == "stack_linear": 
        return StackLinearSquash(input_dim, latent_frame_size, output_dim)
    elif type == 'stack_MLP':
        return StackMLPSquash(input_dim, latent_frame_size, output_dim, hidden_sizes=hidden_sizes)
    elif type == 'stack_transformer':
        return StackTransformerSquash(input_dim, latent_frame_size, output_dim, num_layer=len(hidden_sizes))
    elif type == "stack":
        return Stack(latent_frame_size)
    else:
        raise ValueError("Unknown squasher type")


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
class Stack(nn.Module): #( 128 *2, 4, 128)
    def __init__(self, latent_frame_size): 
        super().__init__()
        self.latent_frame_size = latent_frame_size  # 4


    def forward(self, x):
        B, T, F = x.shape # (BS,32,128)
        # input B, T, F -> B, T // latent_frame_size, F * latent_frame_size
        assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
        T_latent = T // self.latent_frame_size # T_latent = 8
        F_stack = F * self.latent_frame_size # F_stack = 512
        x = x.reshape(B, T_latent, F_stack) # (BS,8,512)
        x = x.view(B, T_latent, -1)
        return x
    
class StackLinearSquash(nn.Module): #( 128 *2, 4, 128)
    def __init__(self, input_dim, latent_frame_size, output_dim): 
        super().__init__()
        self.input_dim = input_dim # 128
        self.latent_frame_size = latent_frame_size  # 4
        self.output_dim = output_dim # 128
        self.linear = nn.Linear(input_dim * latent_frame_size, output_dim) # 128*4
        # print(f'input dim : {self.input_dim * latent_frame_size}')
        
    def forward(self, x):
        B, T, F = x.shape # (BS,32,128)
        # input B, T, F -> B, T // latent_frame_size, F * latent_frame_size
        assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
        T_latent = T // self.latent_frame_size # T_latent = 8
        F_stack = F * self.latent_frame_size # F_stack = 512
        x = x.reshape(B, T_latent, F_stack) # (BS,8,512)
        x = x.view(B * T_latent, F_stack) # (BS*8, 512)
        x = self.linear(x) # (BS*8, 128)
        x = x.view(B, T_latent, -1)
        return x
class StackMLPSquash(nn.Module): 
    def __init__(self, input_dim, latent_frame_size, output_dim, hidden_sizes=[512, 512]): 
        super().__init__()
        self.input_dim = input_dim # 128
        self.latent_frame_size = latent_frame_size  # 4
        self.output_dim = output_dim # 128
        net_list = []
        net_list.append(nn.Linear(input_dim * latent_frame_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)): 
            net_list.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        net_list.append(nn.Linear(hidden_sizes[-1], output_dim))
        self.nets = nn.Sequential(*net_list)
        
    def forward(self, x):
        B, T, F = x.shape # (BS,32,128)
        # input B, T, F -> B, T // latent_frame_size, F * latent_frame_size
        assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
        T_latent = T // self.latent_frame_size # T_latent = 8
        F_stack = F * self.latent_frame_size # F_stack = 512
        x = x.reshape(B, T_latent, F_stack) # (BS,8,512)
        x = x.view(B * T_latent, F_stack) # (BS*8, 512)
        
        for net in self.nets:
            x = net(x)
            if net != self.nets[-1]:
                x = torch.nn.functional.relu(x)
        
        x = x.view(B, T_latent, -1)
        return x
    
class StackTransformerSquash(nn.Module): 
    def __init__(self, input_dim, latent_frame_size, output_dim, num_layer): 
        super().__init__()
        self.input_dim = input_dim # 128
        self.latent_frame_size = latent_frame_size  # 4
        self.output_dim = output_dim # 128
        self.pre_linear = nn.Linear(input_dim*latent_frame_size, output_dim)
        self.attention_mask = init_alibi_biased_mask_future(
            8, 600)# as max len is not that important in alibi
        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=output_dim, 
                    nhead=8, dim_feedforward=output_dim,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True
        )        
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        self.post_linear = nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        B, T, F = x.shape # (BS,32,128)
        # input B, T, F -> B, T // latent_frame_size, F * latent_frame_size
        assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
        T_latent = T // self.latent_frame_size # T_latent = 8
        F_stack = F * self.latent_frame_size # F_stack = 512
        x = x.reshape(B, T_latent, F_stack) # (BS,8,512)
        # x = x.view(B * T_latent, F_stack) # (BS*8, 512)
        
        x = self.pre_linear(x) # (BS*8, 128)
        # x = x.view(B, T_latent, -1) # (BS,8,128)
        B, T = x.shape[:2]
        mask = self.attention_mask[:, :T, :T].clone().detach().to(device=x.device)
        if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
            mask = mask.repeat(x.shape[0], 1, 1)
        x = self.encoder(x, mask=mask)  # (BS,8,128)
        x = self.post_linear(x) # (BS,8,128)

        # x = x.view(B, T_latent, -1)
        return x
class LinearEmotionCondition(nn.Module):
    def __init__(self, condition_dim, output_dim):
        super().__init__()
        self.map = nn.Linear(condition_dim, output_dim)

    def forward(self, sample):
        return self.map(sample)



class DEMOTE_VQVAE(nn.Module) :
    def __init__(self, EMOTE_config, FLINT_config, DEE_config, FLINT_ckpt, DEE, 
                 load_motion_prior=True, output_logits=False) :
        super(DEMOTE_VQVAE, self).__init__()
        
        self.output_logits = output_logits
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
        self.sequence_decoder = BertPriorDecoder(decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior, output_logits=output_logits)
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

    def forward(self, audio_content, condition, audio_emotion=None, sample=False, control_logvar=None, tau=0.1) :
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
        
        output = self.sequence_decoder(condition, audio_emotion, audio_embedding, sample=sample, control_logvar=control_logvar, tau=tau) # (BS,128,53)
        return output # (BS,128,53)


class BertPriorDecoder(nn.Module):
    def __init__(self, decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior, output_logits=False):
        super(BertPriorDecoder, self).__init__()
        self.output_logits = output_logits
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
                # self.motion_prior = TVAE(FLINT_config).motion_decoder
        self.motion_prior = VQVAE2(FLINT_config)
        # Assuming self.motion_prior is a PyTorch model
        if load_motion_prior :
            print(f'Load FLINT checkpoints from {FLINT_ckpt}')
            decoder_ckpt = torch.load(FLINT_ckpt)
            if 'state_dict' in decoder_ckpt:
                decoder_ckpt = decoder_ckpt['state_dict']
            self.motion_prior.load_state_dict(decoder_ckpt)
        
        # freeze decoder
        self.motion_prior.eval()
        for param in self.motion_prior.parameters():
            param.requires_grad = False
            
        # Squasher
        if decoder_config['squash_after'] : #(linear stack, 128 *2, 128,3, 4)
            self.squasher_2_b = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 self.motion_prior.quantize_b.num_embeddings, 
                                                 quant_factor = self.motion_prior.encoder_b.quant_factor, 
                                                 latent_frame_size = 2**self.motion_prior.encoder_b.quant_factor,
                                                 hidden_sizes=decoder_config['hidden_sizes'])
        elif decoder_config['squash_before'] :
            raise NotImplementedError("squash_before is not implemented")
            self.squasher_1_b = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'], decoder_config['latent_frame_size_b'])
        else : 
            raise ValueError("Unknown squasher type")

        if decoder_config['squash_after'] : #(linear stack, 128 *2, 128,3, 4)
            self.squasher_2_t = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 self.motion_prior.quantize_t.num_embeddings, 
                                                 quant_factor = self.motion_prior.encoder_b.quant_factor + self.motion_prior.encoder_t.quant_factor,
                                                 latent_frame_size = 2**(self.motion_prior.encoder_b.quant_factor + self.motion_prior.encoder_t.quant_factor),
                                                 hidden_sizes=decoder_config['hidden_sizes'])
        elif decoder_config['squash_before'] :
            raise NotImplementedError("squash_before is not implemented")
            self.squasher_1_t = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'], decoder_config['latent_frame_size_t'])
        else : 
            raise ValueError("Unknown squasher type")

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


    def decode(self, sample, tau=0.1) :
        
        mask = self.biased_mask[:, :sample.shape[1], :sample.shape[1]].clone().detach().to(device=sample.device)
        if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
            mask = mask.repeat(sample.shape[0], 1, 1)
        
        output = self.bert_decoder(sample, mask=mask) # (BS,64,256)
        output = self.decoder(output) # (BS,64,128)
        # output = self.squasher_2(output) # (BS,16,128)
        logits_t = self.squasher_2_t(output) # (BS, T / (q_b*q_t), n_emb) (BS, 4, n_emb)
        logits_b = self.squasher_2_b(output) # (BS, T/(q_b), n_emb) (BS, 8, n_emb)
        if self.output_logits: 
            return logits_t, logits_b       
        else:
            dec = self.motion_prior.differential_logits_to_image(logits_t, logits_b, tau=tau)
            return dec


    def forward(self, condition, audio_emotion, audio_embedding, sample=False, control_logvar=None, tau=0.1) :

        repeat_num = audio_embedding.shape[1]
        audio_emotion_embedding = self.encode_emotion_condition(audio_emotion, sample=sample, control_logvar=control_logvar) # (BS,128)
        actor_condition_onehot = condition[:,11:] # (BS,32)
        emotion_actor_cat = torch.cat([audio_emotion_embedding, actor_condition_onehot], dim=1) # (BS,160)
        style_embedding = self.condition_feature_layer(emotion_actor_cat).unsqueeze(1).repeat(1,repeat_num,1) # (BS,T,128)

        if self.add_condition :# add
            styled_audio_add = audio_embedding + style_embedding
            output = self.decode(styled_audio_add, tau=tau)
        else :# cat
            styled_audio_cat = torch.cat([audio_embedding, style_embedding], dim=-1) # (BS,64,256)
            output = self.decode(styled_audio_cat, tau=tau) # (BS,128,53)

        return output # (BS,128,53)
    
    
class DEMOTE_VQVAE_condition(nn.Module) :
    def __init__(self, EMOTE_config, FLINT_config, DEE_config, FLINT_ckpt, DEE, 
                 load_motion_prior=True, output_logits=False) :
        super(DEMOTE_VQVAE_condition, self).__init__()
        
        self.output_logits = output_logits
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
        self.sequence_decoder = BertPriorDecoder_condition(decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior, output_logits=output_logits)
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

    def forward(self, audio_content, condition, audio_emotion=None, sample=False, control_logvar=None, tau=0.1) :
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
        
        output = self.sequence_decoder(condition, audio_emotion, audio_embedding, sample=sample, control_logvar=control_logvar, tau=tau) # (BS,128,53)
        return output # (BS,128,53)
    
    
    def nondeterministic_forward(self, audio_content, condition, audio_emotion=None, sample=False, control_logvar=None, 
                                 temp_t=0.1, temp_b=0.1) :
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
        
        output = self.sequence_decoder.nondeterministic_forward(condition, audio_emotion, audio_embedding, sample=sample, control_logvar=control_logvar, 
                                                                temp_t=temp_t, temp_b=temp_b) # (BS,128,53)
        return output # (BS,128,53)
class BertPriorDecoder_condition(nn.Module):
    def __init__(self, decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior, output_logits=False):
        super(BertPriorDecoder_condition, self).__init__()
        self.output_logits = output_logits
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
                # self.motion_prior = TVAE(FLINT_config).motion_decoder
        self.motion_prior = VQVAE2(FLINT_config)
        # Assuming self.motion_prior is a PyTorch model
        if load_motion_prior :
            print(f'Load FLINT checkpoints from {FLINT_ckpt}')
            decoder_ckpt = torch.load(FLINT_ckpt)
            if 'state_dict' in decoder_ckpt:
                decoder_ckpt = decoder_ckpt['state_dict']
            self.motion_prior.load_state_dict(decoder_ckpt)
        
        # freeze decoder
        self.motion_prior.eval()
        for param in self.motion_prior.parameters():
            param.requires_grad = False
            
        # Squasher
        if decoder_config['squash_after'] : #(linear stack, 128 *2, 128,3, 4)
            self.squasher_2_b = _create_squasher("stack", 
                                                 decoder_config['feature_dim'], # notused
                                                 self.motion_prior.quantize_b.num_embeddings, #notused
                                                 quant_factor = self.motion_prior.encoder_b.quant_factor, # notused
                                                 latent_frame_size = 2**self.motion_prior.encoder_b.quant_factor# used
                                                 )
            
            self.squasher_2_b_2 = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim']*(2**self.motion_prior.encoder_b.quant_factor) + self.motion_prior.quantize_t.embedding_dim, 
                                                 self.motion_prior.quantize_b.num_embeddings, # output_dim
                                                 quant_factor = self.motion_prior.encoder_b.quant_factor, 
                                                 latent_frame_size = 1,
                                                 hidden_sizes=decoder_config['hidden_sizes'])
        elif decoder_config['squash_before'] :
            raise NotImplementedError("squash_before is not implemented")
            self.squasher_1_b = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'], decoder_config['latent_frame_size_b'])
        else : 
            raise ValueError("Unknown squasher type")

        if decoder_config['squash_after'] : #(linear stack, 128 *2, 128,3, 4)
            self.squasher_2_t = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 self.motion_prior.quantize_t.num_embeddings, 
                                                 quant_factor = self.motion_prior.encoder_b.quant_factor + self.motion_prior.encoder_t.quant_factor,
                                                 latent_frame_size = 2**(self.motion_prior.encoder_b.quant_factor + self.motion_prior.encoder_t.quant_factor),
                                                 hidden_sizes=decoder_config['hidden_sizes'])
        elif decoder_config['squash_before'] :
            raise NotImplementedError("squash_before is not implemented")
            self.squasher_1_t = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'], decoder_config['latent_frame_size_t'])
        else : 
            raise ValueError("Unknown squasher type")

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
        max_len = self.DEE_audio.audio_max_seq_len
        
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


    def decode(self, sample, tau=0.1) :
        
        mask = self.biased_mask[:, :sample.shape[1], :sample.shape[1]].clone().detach().to(device=sample.device)
        if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
            mask = mask.repeat(sample.shape[0], 1, 1)
        
        output = self.bert_decoder(sample, mask=mask) # (BS,64,256)
        output = self.decoder(output) # (BS,64,128)
        # output = self.squasher_2(output) # (BS,16,128)
        logits_t = self.squasher_2_t(output) # (BS, T / (q_b*q_t), n_emb) (BS, 4, n_emb)
        onehot_t = self.motion_prior.differential_logit_to_onehot(logits_t,tau)
        quant_t = torch.matmul(onehot_t, self.motion_prior.quantize_t.codebook.weight)
        dec_t = self.motion_prior.decoder_t(quant_t)
        stacked_features = self.squasher_2_b(output) # (BS, T/(q_b), n_emb) (BS, 8, n_emb)
        enc_b = torch.cat([stacked_features, dec_t], dim=2)
        logits_b = self.squasher_2_b_2(enc_b) # (BS, T/(q_b), n_emb)
        
        if self.output_logits: 
            return logits_t, logits_b
        else:
            # dec = self.motion_prior.differential_logits_to_image(logits_t, logits_b, tau=tau)
            onehot_b = self.motion_prior.differential_logit_to_onehot(logits_b,tau)
            quant_b = torch.matmul(onehot_b, self.motion_prior.quantize_b.codebook.weight)
            dec = self.motion_prior.decode(quant_t, quant_b)
            return dec


    def forward(self, condition, audio_emotion, audio_embedding, sample=False, control_logvar=None, tau=0.1) :

        repeat_num = audio_embedding.shape[1]
        audio_emotion_embedding = self.encode_emotion_condition(audio_emotion, sample=sample, control_logvar=control_logvar) # (BS,128)
        actor_condition_onehot = condition[:,11:] # (BS,32)
        emotion_actor_cat = torch.cat([audio_emotion_embedding, actor_condition_onehot], dim=1) # (BS,160)
        style_embedding = self.condition_feature_layer(emotion_actor_cat).unsqueeze(1).repeat(1,repeat_num,1) # (BS,T,128)

        if self.add_condition :# add
            styled_audio_add = audio_embedding + style_embedding
            output = self.decode(styled_audio_add, tau=tau)
        else :# cat
            styled_audio_cat = torch.cat([audio_embedding, style_embedding], dim=-1) # (BS,64,256)
            output = self.decode(styled_audio_cat, tau=tau) # (BS,128,53)

        return output # (BS,128,53)
    
    def nondeterministic_forward(self,condition, audio_emotion, audio_embedding, sample=False, control_logvar=None, 
                                 temp_t=0.1, temp_b=0.1) :

        repeat_num = audio_embedding.shape[1]
        audio_emotion_embedding = self.encode_emotion_condition(audio_emotion, sample=sample, control_logvar=control_logvar) # (BS,128)
        actor_condition_onehot = condition[:,11:] # (BS,32)
        emotion_actor_cat = torch.cat([audio_emotion_embedding, actor_condition_onehot], dim=1) # (BS,160)
        style_embedding = self.condition_feature_layer(emotion_actor_cat).unsqueeze(1).repeat(1,repeat_num,1) # (BS,T,128)

        if self.add_condition :# add
            styled_audio_add = audio_embedding + style_embedding
            output = self.nondeterministic_decode(styled_audio_add, temp_t=temp_t, temp_b=temp_b)
        else :# cat
            styled_audio_cat = torch.cat([audio_embedding, style_embedding], dim=-1) # (BS,64,256)
            output = self.nondeterministic_decode(styled_audio_cat, temp_t=temp_t, temp_b=temp_b) # (BS,128,53)

        return output # (BS,128,53)
    
    def nondeterministic_decode(self, sample, temp_t=0.1, temp_b=0.1) :
        
        mask = self.biased_mask[:, :sample.shape[1], :sample.shape[1]].clone().detach().to(device=sample.device)
        if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
            mask = mask.repeat(sample.shape[0], 1, 1)
        
        output = self.bert_decoder(sample, mask=mask) # (BS,64,256)
        output = self.decoder(output) # (BS,64,128)
        # output = self.squasher_2(output) # (BS,16,128)
        logits_t = self.squasher_2_t(output) # (BS, T / (q_b*q_t), n_emb) (BS, 4, n_emb)
        # onehot_t = self.motion_prior.differential_logit_to_onehot(logits_t,tau)
        logits_t = logits_t / temp_t
        prob_t = torch.nn.functional.softmax(logits_t, dim= -1)
        B, T, n_emb = prob_t.shape
        prob_t = prob_t.reshape(B*T, n_emb)
        ix = torch.multinomial(prob_t, num_samples=1).squeeze().reshape(B,T) # (B*T, 1) -> (B*T) -> (B,T)
        onehot_t = torch.nn.functional.one_hot(ix, num_classes = self.motion_prior.quantize_t.num_embeddings) # (B, T ,256)
        quant_t = torch.matmul(onehot_t.float(), self.motion_prior.quantize_t.codebook.weight)
        
        dec_t = self.motion_prior.decoder_t(quant_t)
        stacked_features = self.squasher_2_b(output) # (BS, T/(q_b), n_emb) (BS, 8, n_emb)
        enc_b = torch.cat([stacked_features, dec_t], dim=2)
        logits_b = self.squasher_2_b_2(enc_b) # (BS, T/(q_b), n_emb)
        
        if self.output_logits: 
            return logits_t, logits_b
        else:
            # dec = self.motion_prior.differential_logits_to_image(logits_t, logits_b, tau=tau)
            # onehot_b = self.motion_prior.differential_logit_to_onehot(logits_b,tau)
            # quant_b = torch.matmul(onehot_b, self.motion_prior.quantize_b.codebook.weight)
            logits_b = logits_b / temp_b
            prob_b = torch.nn.functional.softmax(logits_b, dim= -1)
            B, T, n_emb = prob_b.shape
            prob_b = prob_b.reshape(B*T, n_emb)
            ix = torch.multinomial(prob_b, num_samples=1).squeeze().reshape(B,T) # (B*T, 1) -> (B*T) -> (B,T)
            onehot_b = torch.nn.functional.one_hot(ix, num_classes = self.motion_prior.quantize_b.num_embeddings) # (B, T ,256)
            quant_b = torch.matmul(onehot_b.float(), self.motion_prior.quantize_b.codebook.weight)
            dec = self.motion_prior.decode(quant_t, quant_b)
            return dec
class DEMOTE_codetalker(nn.Module) :
    def __init__(self, EMOTE_config, FLINT_config, DEE_config, FLINT_ckpt, DEE, 
                 load_motion_prior=True, output_logits=False) :
        super(DEMOTE_codetalker, self).__init__()
        
        self.output_logits = output_logits
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
        self.sequence_decoder = BertPriorDecoder_codetalker(decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior, output_logits=output_logits)
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

    def forward(self, audio_content, condition, audio_emotion=None, sample=False, control_logvar=None, tau=0.1) :
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
        
        output = self.sequence_decoder(condition, audio_emotion, audio_embedding, sample=sample, control_logvar=control_logvar, tau=tau) # (BS,128,53)
        return output # (BS,128,53)


class BertPriorDecoder_codetalker(nn.Module):
    def __init__(self, decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior, output_logits=False):
        super(BertPriorDecoder_codetalker, self).__init__()
        self.output_logits = output_logits
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
                # self.motion_prior = TVAE(FLINT_config).motion_decoder
        self.motion_prior = VQVAE2(FLINT_config)
        # Assuming self.motion_prior is a PyTorch model
        if load_motion_prior :
            print(f'Load FLINT checkpoints from {FLINT_ckpt}')
            decoder_ckpt = torch.load(FLINT_ckpt)
            if 'state_dict' in decoder_ckpt:
                decoder_ckpt = decoder_ckpt['state_dict']
            self.motion_prior.load_state_dict(decoder_ckpt)
        
        # freeze decoder
        self.motion_prior.eval()
        for param in self.motion_prior.parameters():
            param.requires_grad = False
            
        # Squasher
        if decoder_config['squash_after'] : #(linear stack, 128 *2, 128,3, 4)
            self.squasher_2_b = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 self.motion_prior.quantize_b.embedding_dim, 
                                                 quant_factor = self.motion_prior.encoder_b.quant_factor, 
                                                 latent_frame_size = 2**self.motion_prior.encoder_b.quant_factor,
                                                 hidden_sizes=decoder_config['hidden_sizes'])
        elif decoder_config['squash_before'] :
            raise NotImplementedError("squash_before is not implemented")
            self.squasher_1_b = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'], decoder_config['latent_frame_size_b'])
        else : 
            raise ValueError("Unknown squasher type")

        if decoder_config['squash_after'] : #(linear stack, 128 *2, 128,3, 4)
            self.squasher_2_t = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 self.motion_prior.quantize_t.embedding_dim, 
                                                 quant_factor = self.motion_prior.encoder_b.quant_factor + self.motion_prior.encoder_t.quant_factor,
                                                 latent_frame_size = 2**(self.motion_prior.encoder_b.quant_factor + self.motion_prior.encoder_t.quant_factor),
                                                 hidden_sizes=decoder_config['hidden_sizes'])
        elif decoder_config['squash_before'] :
            raise NotImplementedError("squash_before is not implemented")
            self.squasher_1_t = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'], decoder_config['latent_frame_size_t'])
        else : 
            raise ValueError("Unknown squasher type")

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


    def decode(self, sample, tau=0.1) :
        
        mask = self.biased_mask[:, :sample.shape[1], :sample.shape[1]].clone().detach().to(device=sample.device)
        if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
            mask = mask.repeat(sample.shape[0], 1, 1)
        
        output = self.bert_decoder(sample, mask=mask) # (BS,64,256)
        output = self.decoder(output) # (BS,64,128)
        # output = self.squasher_2(output) # (BS,16,128)
        z_t = self.squasher_2_t(output) # (BS, T / (q_b*q_t), e_dim) (BS, 4, e_dim)
        z_b = self.squasher_2_b(output) # (BS, T/(q_b), e_dim) (BS, 8, e_dim)
        
        return z_t, z_b


    def forward(self, condition, audio_emotion, audio_embedding, sample=False, control_logvar=None, tau=0.1) :

        repeat_num = audio_embedding.shape[1]
        audio_emotion_embedding = self.encode_emotion_condition(audio_emotion, sample=sample, control_logvar=control_logvar) # (BS,128)
        actor_condition_onehot = condition[:,11:] # (BS,32)
        emotion_actor_cat = torch.cat([audio_emotion_embedding, actor_condition_onehot], dim=1) # (BS,160)
        style_embedding = self.condition_feature_layer(emotion_actor_cat).unsqueeze(1).repeat(1,repeat_num,1) # (BS,T,128)

        if self.add_condition :# add
            styled_audio_add = audio_embedding + style_embedding
            output = self.decode(styled_audio_add, tau=tau)
        else :# cat
            styled_audio_cat = torch.cat([audio_embedding, style_embedding], dim=-1) # (BS,64,256)
            output = self.decode(styled_audio_cat, tau=tau) # (BS,128,53)

        return output # (BS,128,53)
    
    
class DEMOTE_vanila_VQVAE(nn.Module) :
    def __init__(self, EMOTE_config, FLINT_config, DEE_config, FLINT_ckpt, DEE, 
                 load_motion_prior=True, output_logits=False) :
        super(DEMOTE_vanila_VQVAE, self).__init__()
        
        self.output_logits = output_logits
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
        self.sequence_decoder = BertPriorDecoder_vanila_VQVAE(decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior, output_logits=output_logits)
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

    def forward(self, audio_content, condition, audio_emotion=None, sample=False, control_logvar=None, tau=0.1) :
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
        
        output = self.sequence_decoder(condition, audio_emotion, audio_embedding, sample=sample, control_logvar=control_logvar, tau=tau) # (BS,128,53)
        return output # (BS,128,53)


class BertPriorDecoder_vanila_VQVAE(nn.Module):
    def __init__(self, decoder_config, FLINT_config, FLINT_ckpt, DEE, load_motion_prior, output_logits=False):
        super(BertPriorDecoder_vanila_VQVAE, self).__init__()
        self.output_logits = output_logits
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
                # self.motion_prior = TVAE(FLINT_config).motion_decoder
        self.motion_prior = VQVAE(FLINT_config, version='vanila')
        # Assuming self.motion_prior is a PyTorch model
        if load_motion_prior :
            print(f'Load FLINT checkpoints from {FLINT_ckpt}')
            decoder_ckpt = torch.load(FLINT_ckpt)
            if 'state_dict' in decoder_ckpt:
                decoder_ckpt = decoder_ckpt['state_dict']
            self.motion_prior.load_state_dict(decoder_ckpt)
        
        # freeze decoder
        self.motion_prior.eval()
        for param in self.motion_prior.parameters():
            param.requires_grad = False
            
        # Squasher
        if decoder_config['squash_after'] : #(linear stack, 128 *2, 128,3, 4)
            self.squasher_2 = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 self.motion_prior.quantize.num_embeddings, 
                                                 quant_factor = self.motion_prior.encoder.quant_factor, 
                                                 latent_frame_size = 2**self.motion_prior.encoder.quant_factor,
                                                 hidden_sizes=decoder_config['hidden_sizes'])
        elif decoder_config['squash_before'] :
            raise NotImplementedError("squash_before is not implemented")
            self.squasher_1 = _create_squasher(decoder_config['squash_type'], decoder_config['feature_dim'], 
                                                 decoder_config['feature_dim']*dim_factor, decoder_config['quant_factor'], decoder_config['latent_frame_size_b'])
        else : 
            raise ValueError("Unknown squasher type")

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


    def decode(self, sample, tau=0.1) :
        
        mask = self.biased_mask[:, :sample.shape[1], :sample.shape[1]].clone().detach().to(device=sample.device)
        if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
            mask = mask.repeat(sample.shape[0], 1, 1)
        
        output = self.bert_decoder(sample, mask=mask) # (BS,64,256)
        output = self.decoder(output) # (BS,64,128)
        logits = self.squasher_2(output) # (BS, T / (q_b*q_t), n_emb) (BS, 4, n_emb)
        if self.output_logits: 
            return logits     
        else:
            dec = self.motion_prior.differential_logits_to_image(logits, tau=tau)
            return dec


    def forward(self, condition, audio_emotion, audio_embedding, sample=False, control_logvar=None, tau=0.1) :

        repeat_num = audio_embedding.shape[1]
        audio_emotion_embedding = self.encode_emotion_condition(audio_emotion, sample=sample, control_logvar=control_logvar) # (BS,128)
        actor_condition_onehot = condition[:,11:] # (BS,32)
        emotion_actor_cat = torch.cat([audio_emotion_embedding, actor_condition_onehot], dim=1) # (BS,160)
        style_embedding = self.condition_feature_layer(emotion_actor_cat).unsqueeze(1).repeat(1,repeat_num,1) # (BS,T,128)

        if self.add_condition :# add
            styled_audio_add = audio_embedding + style_embedding
            output = self.decode(styled_audio_add, tau=tau)
        else :# cat
            styled_audio_cat = torch.cat([audio_embedding, style_embedding], dim=-1) # (BS,64,256)
            output = self.decode(styled_audio_cat, tau=tau) # (BS,128,53)

        return output # (BS,128,53)