import torch 
from torch import nn
import math
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from omegaconf import DictConfig, OmegaConf, open_dict
from .MotionPrior import MotionPrior

# from inferno.models.temporal.motion_prior.L2lMotionPrior import L2lVqVae, create_squasher
    
from ..utils.extra import get_checkpoint_with_kwargs
import sys

from .base_faceformer import init_faceformer_biased_mask, init_faceformer_biased_mask_future, init_mask, init_mask_future

from ..utils.extra import class_from_str, get

def positional_encoding_from_cfg(cfg, feature_dim= None):
    if feature_dim is None:
        feature_dim = cfg['feature_dim ']
    if cfg['type'] == 'PeriodicPositionalEncoding': 
        # return PeriodicPositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
        return PeriodicPositionalEncoding(feature_dim, **cfg)
    elif cfg['type'] == 'PositionalEncoding':
        # return PositionalEncoding(cfg.feature_dim, **cfg.positional_encoding)
        return PositionalEncoding(feature_dim, **cfg)
    elif cfg['type'] == 'LearnedPositionEmbedding':
        return LearnedPositionEmbedding(cfg.max_seq_len, feature_dim)
    elif not cfg['type'] or str(cfg['type']).lower() == 'none':
        return None
    raise ValueError("Unsupported positional encoding")

def load_motion_prior_net(path, trainable=False):
    from pathlib import Path
    from inferno.utils.other import get_path_to_assets

    path = Path(path)
    if not path.is_absolute():
        path = get_path_to_assets() / path

    with open(path / "cfg.yaml", 'r') as f:
        model_config = OmegaConf.load(f)
    checkpoint_mode = 'best' # resuming in the same stage, we want to pick up where we left of
    
    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(
        model_config, "", 
        checkpoint_mode=checkpoint_mode,
        pattern="val"
        )


    motion_prior_net_class = class_from_str(model_config.model.pl_module_class, sys.modules[__name__])
    motion_prior_net = motion_prior_net_class.instantiate(model_config, "", "", checkpoint, checkpoint_kwargs)
    if not trainable:
        motion_prior_net.eval()
        # freeze model
        for p in motion_prior_net.parameters():
            p.requires_grad = False
    return motion_prior_net

def style_from_cfg(cfg):
    style_type = get(cfg,'style_embedding', 'onehot_linear')
    if isinstance(style_type, (DictConfig, dict)): 
        # new preferred way
        style_cfg = style_type
        style_type = style_cfg['type']
        if style_type == 'emotion_linear' or style_type == 'video_emotion_linear':
            if style_cfg['use_shape']: 
                style_cfg['shape_dim'] = cfg['shape']['n_shape']
                # with open_dict(style_cfg) as c:
                #     c.shape_dim = cfg.flame.n_shape
            return LinearEmotionCondition(style_cfg, output_dim=cfg['feature_dim'])
        elif style_type in ['onehot_linear', 'onehot_identity_linear']:
            return OneHotIdentityCondition(style_cfg, output_dim=cfg['feature_dim'])
        raise ValueError(f"Unsupported style embedding type '{style_type}'")

    # old way (for backwards compatibility)
    if style_type == 'onehot_linear':
        return nn.Linear(cfg['num_training_subjects'], cfg['feature_dim'], bias=False)
    elif style_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported style embedding type '{style_type}'")

class StyleConditioning(torch.nn.Module): 

    def __init__(self): 
        super().__init__()


    def forward(self, sample, **kwargs): 
        raise NotImplementedError("Subclasses must implement this method")

class FeedForwardDecoder(nn.Module): 

    """
    A base-class for feed-forward (non-autoregressive) decoders.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.style_type = get(cfg,'style_embedding', 'onehot_linear')
        self.obj_vector = style_from_cfg(cfg)
        self.style_op = get(cfg,'style_op', 'add')
        self.PE = positional_encoding_from_cfg(cfg['positional_encoding'], cfg['feature_dim'] )
        self.cfg = cfg

    def forward(self, sample, train=False, teacher_forcing=True): 
        sample["hidden_feature"] = sample["seq_encoder_output"]
        hidden_states = self._positional_enc(sample)

        styled_hidden_states = self._style(sample, hidden_states)

        decoded_offsets = self._decode(sample, styled_hidden_states)

        sample["predicted_vertices"] = decoded_offsets
        sample = self._post_prediction(sample)
        return sample

    def _positional_enc(self, sample): 
        hidden_states = sample["hidden_feature"] 
        if self.PE is not None:
            hidden_states = self.PE(hidden_states)
        return hidden_states

    def _style_dim_factor(self): 
        if self.obj_vector is None: 
            return 1
        elif self.style_op == "cat":
            return 2
        elif self.style_op in ["add", "none", "style_only"]:
            return 1
        raise ValueError(f"Invalid operation: '{self.style_op}'")
       
    def _pe_dim_factor(self):
        dim_factor = 1
        if self.PE is not None: 
            dim_factor = self.PE.output_size_factor()
        return dim_factor

    def _total_dim_factor(self): 
        return self._style_dim_factor() * self._pe_dim_factor()

    def _style(self, sample, hidden_states): 
        if self.obj_vector is None:
            return hidden_states
        if isinstance(self.obj_vector, StyleConditioning):
            # the new way
            style_emb = self.obj_vector(sample)
            # return style_emb
        else:
            # backwards compatibility
            one_hot = sample["one_hot"]
            obj_embedding = self.obj_vector(one_hot)
            style_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
        if self.style_op == "add":
            styled_hidden_states = hidden_states + style_emb
        elif self.style_op == "cat":
            if style_emb.ndim == 2:
                style_emb = style_emb.unsqueeze(1)
            if style_emb.shape[1] == 1:
                style_emb = style_emb.repeat(1, hidden_states.shape[1], 1)
            styled_hidden_states = torch.cat([hidden_states, style_emb], dim=-1)
        elif self.style_op == "none": # no style, for debugging purposes only
            styled_hidden_states = hidden_states
        elif self.style_op == "style_only": # no hidden features, only style. for debugging purposes only
            styled_hidden_states = style_emb
        else: 
            raise ValueError(f"Invalid operation: '{self.style_op}'")
        
        return styled_hidden_states

    def _decode(self, sample, hidden_states) :
        """
        Hiddent states are the ouptut of the encoder
        Returns a tensor of vertex offsets.
        """
        raise NotImplementedError("The subdclass must implement this")

    def _post_prediction(self, sample):
        """
        Adds the template vertices to the predicted offsets
        """
        template = sample["template"]
        vertices_out = sample["predicted_vertices"]
        vertices_out = vertices_out + template[:, None, ...]
        sample["predicted_vertices"] = vertices_out
        return sample

    def get_trainable_parameters(self): 
        return [p for p in self.parameters() if p.requires_grad]

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

class StackLinearSquash(nn.Module): 
    def __init__(self, input_dim, latent_frame_size, output_dim): 
        super().__init__()
        self.input_dim = input_dim
        self.latent_frame_size = latent_frame_size
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim * latent_frame_size, output_dim)
        
    def forward(self, x):
        B, T, F = x.shape
        # input B, T, F -> B, T // latent_frame_size, F * latent_frame_size
        assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
        T_latent = T // self.latent_frame_size
        F_stack = F * self.latent_frame_size
        x = x.reshape(B, T_latent, F_stack)
        x = x.view(B * T_latent, F_stack)
        x = self.linear(x)
        x = x.view(B, T_latent, -1)
        return x

class BertPriorDecoder(FeedForwardDecoder):
    """
    A decoder that uses a transformer encoder and a motion prior network to decode the FLAME parameters
    """
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        dim_factor = self._total_dim_factor()

        ## encode style, audio
        if cfg['num_layers'] > 0:
            encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=cfg['feature_dim'] * dim_factor, 
                        nhead=cfg['nhead'], dim_feedforward=dim_factor*cfg['feature_dim'], 
                        activation=cfg['activation'],
                        dropout=cfg['dropout'], batch_first=True
            )        
            self.bert_decoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg['num_layers'])
        else:
            self.bert_decoder = None
        
        self.post_bug_fix = get(cfg,'post_bug_fix', False)

        self.temporal_bias_type = get(cfg,'temporal_bias_type', 'none')
        # max_len = cfg.max_len
        max_len = 1200
        if self.temporal_bias_type == 'faceformer':
            self.biased_mask = init_faceformer_biased_mask(num_heads = cfg['nhead'], max_seq_len = max_len, period=cfg['period'])
        elif self.temporal_bias_type == 'faceformer_future':
            self.biased_mask = init_faceformer_biased_mask_future(num_heads = cfg['nhead'], max_seq_len = max_len, period=cfg['period'])
        elif self.temporal_bias_type == 'classic':
            self.biased_mask = init_mask(num_heads = cfg['nhead'], max_seq_len = max_len)
        elif self.temporal_bias_type == 'classic_future':
            self.biased_mask = init_mask_future(num_heads = cfg['nhead'], max_seq_len = max_len)
        elif self.temporal_bias_type == 'none':
            self.biased_mask = None
        else:
            raise ValueError(f"Unsupported temporal bias type '{self.temporal_bias_type}'")

        ## Pretrained motion prior
        self.motion_prior : MotionPrior = load_motion_prior_net(cfg['motion_prior']['path'], get(cfg['motion_prior'],'trainable', False))
        self.latent_frame_size = self.motion_prior.latent_frame_size()
        quant_factor = self.motion_prior.quant_factor()
        bottleneck_dim = self.motion_prior.bottleneck_dim()
        self.motion_prior.discard_encoder()
        self.flame = self.motion_prior.get_flame()
        
        ## SQUASH
        # temporal downsampling (if need be) to match the motion prior network latent frame rate
        if get(cfg,'squash_before',True) :
            self.squasher = self._create_squasher(get(cfg,'squash_type','conv'), cfg['feature_dim'] * dim_factor, cfg['feature_dim'] * dim_factor, quant_factor)
        else:
            self.squasher = None

        self.decoder = nn.Linear(dim_factor*cfg['feature_dim'], bottleneck_dim)

        # trying init to prevent the loss from exploding in the beginning
        nn.init.constant_(self.decoder.weight, 0)
        nn.init.constant_(self.decoder.bias, 0)

        # self.bottleneck_proj = nn.Linear(cfg.feature_dim * dim_factor, bottleneck_dim)

        if get(cfg,'squash_after', False):
            self.squasher_2 = self._create_squasher(get(cfg,'squash_type', 'conv'), bottleneck_dim, bottleneck_dim, quant_factor)
        else:
            self.squasher_2 = None

        assert not (self.squasher is not None and self.squasher_2 is not None), "Cannot have two squashers"

    def _create_squasher(self, type, input_dim, output_dim, quant_factor): 
        if type == "conv": 
            return ConvSquasher(input_dim, quant_factor, output_dim)
        elif type == "stack_linear": 
            return StackLinearSquash(input_dim, self.latent_frame_size, output_dim)
        else: 
            raise ValueError("Unknown squasher type")

    def train(self, mode: bool = True) -> "BertPriorDecoder":
        super().train(mode)
        if get(self.cfg['motion_prior'],'trainable', False):
            self.motion_prior.train(mode)
        else: 
            self.motion_prior.eval()
        return self

    def to(self, device):
        super().to(device)
        if self.biased_mask is not None:
            self.biased_mask = self.biased_mask.to(device)
        if self.bert_decoder is not None:
            self.bert_decoder.to(device)
        self.motion_prior.to(device)
        if self.squasher is not None:
            self.squasher.to(device)
        if self.squasher_2 is not None:
            self.squasher_2.to(device)
        self.decoder.to(device)
        # self.bottleneck_proj.to(device
        return self

    def _rotation_representation(self):
        return self.motion_prior._rotation_representation()

    def forward(self, sample, train=False, teacher_forcing=True): 
        sample = super().forward(sample, train=train, teacher_forcing=teacher_forcing)
        return sample

    def get_shape_model(self):
        return self.motion_prior.get_flame() 

    def decoder_output_dim(self):
        return self.cfg['vertices_dim']

    def _post_prediction(self, sample):
        """
        Overrides the base class method to apply the motion prior network. 
        """
        sample = self._apply_motion_prior(sample)
        super()._post_prediction(sample)
        return sample

    def _apply_motion_prior(self, sample):
        decoded_offsets = sample["predicted_vertices"]
        B,T = decoded_offsets.shape[:2]
        batch = {}
        batch[self.motion_prior.input_key_for_decoding_step()] = decoded_offsets 
        T_real = sample["gt_vertices"].shape[1] if "gt_vertices" in sample.keys() else sample["processed_audio"].shape[1]

        T_motion_prior = (T_real // self.latent_frame_size) * self.latent_frame_size
        if T_motion_prior < T_real:
            T_padded = int(math.ceil(T_real / self.latent_frame_size) * self.latent_frame_size)

            # pad to the nearest multiple of the motion prior latent frame size along the temporal dimension T
            # B, T, C -> B, T_padded, C 
            if self.squasher is not None:
                # we're already working on the latent framize size, the data has already been squashed
                padding_size = T_padded // self.latent_frame_size - T_real // self.latent_frame_size
            elif self.squasher_2 is not None:
                # we're working on the original frame size, the data has not been squashed yet
                padding_size = T_padded - T_real 
            batch[self.motion_prior.input_key_for_decoding_step()] = torch.nn.functional.pad(
                decoded_offsets, (0,0, 0, padding_size), mode='constant', value=0
            )

            if self.squasher is not None:
                assert batch[self.motion_prior.input_key_for_decoding_step()].shape[1] == T_padded // self.latent_frame_size, \
                    f"{batch[self.motion_prior.input_key_for_decoding_step()].shape[1]} != {T_padded // self.latent_frame_size}"
            elif self.squasher_2 is not None:
                assert batch[self.motion_prior.input_key_for_decoding_step()].shape[1] == T_padded, \
                    f"{batch[self.motion_prior.input_key_for_decoding_step()].shape[1]} != {T_padded}"

        if self.squasher_2 is not None:
            batch[self.motion_prior.input_key_for_decoding_step()] = self.squasher_2(batch[self.motion_prior.input_key_for_decoding_step()])

        if "gt_shape" in sample:
            batch["gt_shape"] = sample["gt_shape"]
        if "template" in sample:
            batch["template"] = sample["template"]
        if "gt_tex" in sample:
            batch["gt_tex"] = sample["gt_tex"]
        
        sample["prior_input_sequence"] = batch[self.motion_prior.input_key_for_decoding_step()]

        batch = self.motion_prior.decoding_step(batch)

        # if T_real < T:
        # remove the padding
        for i, key in enumerate(self.motion_prior.cfg.model.sequence_components.keys()):
            batch["reconstructed_" + key] = batch["reconstructed_" + key][:,:T_real]
        batch["reconstructed_vertices"] = batch["reconstructed_vertices"][:,:T_real]
        
        for i, key in enumerate(self.motion_prior.cfg.model.sequence_components.keys()):
            sample["predicted_" + key] = batch["reconstructed_" + key]
        sample["predicted_vertices"] = batch["reconstructed_vertices"]
        

        # compute the offsets from neutral, that's how the output of "predicted_vertices" is expected to be
        if "gt_shape" not in sample.keys():
            template = sample["template"]
            assert B == 1, "Batch size must be 1 if we want to change the template inside FLAME"
            flame_template = self.flame.v_template
            self.flame.v_template = template.squeeze(1).view(-1, 3)
            shape_params = torch.zeros((B,T_real, self.flame.cfg.n_shape), device=sample["predicted_vertices"].device)
        else: 
            flame_template = None
            # sample["gt_shape"].shape = (B, n_shape) -> add T dimension and repeat
            if len(sample["gt_shape"].shape) == 2:
                shape_params = sample["gt_shape"][:, None, ...].repeat(1, T_real, 1)
            elif len(sample["gt_shape"].shape) == 3:
                shape_params = sample["gt_shape"]

        B = sample["predicted_vertices"].shape[0]
        vertices_neutral = self._neutral_shape(B, sample["predicted_exp"].shape[-1], shape_params=shape_params)
        # vertices_neutral = vertices_neutral.expand(B, T_real, -1, -1)
        vertex_offsets = sample["predicted_vertices"] - vertices_neutral # compute the offset that is then added to the template shape
        sample["predicted_vertices"] = vertex_offsets

        if flame_template is not None:  # correct the flame template back if need be
            self.flame.v_template = flame_template
        return sample

    def _neutral_shape(self, B, exp_dim, shape_params): 
        with torch.no_grad():
            # vertice_neutral, _, _ = self.flame.forward(shape_params[0:1, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
                        # vertice_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], torch.zeros_like(expression_params[0:1, ...])) # compute neutral shape
            zero_exp = torch.zeros((B, exp_dim), dtype=shape_params.dtype, device=shape_params.device)
            # compute neutral shape for each batch (but not each frame, unnecessary)
            vertices_neutral, _, _ = self.flame.forward(shape_params[:, 0, ...], zero_exp) # compute neutral shape
            vertices_neutral = vertices_neutral.contiguous().view(vertices_neutral.shape[0], -1)[:, None, ...]
        return vertices_neutral

    def _decode(self, sample, styled_hidden_states):
        B, T, F = styled_hidden_states.shape
        # # BTF to BFT (for 1d conv)
        # if self.cfg.get('squash_before', True):
        if self.squasher is not None:
            styled_hidden_states = self.squasher(styled_hidden_states)

        if self.bert_decoder is not None:
            if self.biased_mask is not None: 
                mask = self.biased_mask[:, :styled_hidden_states.shape[1], :styled_hidden_states.shape[1]].clone() \
                    .detach().to(device=styled_hidden_states.device)
                if mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                    mask = mask.repeat(styled_hidden_states.shape[0], 1, 1)
            else: 
                mask = None
            decoded_offsets = self.bert_decoder(styled_hidden_states, mask=mask)
        else: 
            decoded_offsets = styled_hidden_states

        B, T = decoded_offsets.shape[:2]
        decoded_offsets = decoded_offsets.view(B*T, -1)
        ## INSANE BUG WARNING (passing in styled_hidden_states instead of decoded_offsets)
        if not self.post_bug_fix:
            decoded_offsets = self.decoder(styled_hidden_states)
        ## END INSANE BUG WARNING
        ## BUG FIX
        else:
            decoded_offsets = self.decoder(decoded_offsets)
        ## END BUG FIX
        decoded_offsets = decoded_offsets.view(B, T, -1) 
        return decoded_offsets