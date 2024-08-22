import torch 
from torch import Tensor
import sys
from .Bases import SequenceEncoder

class LinearSequenceEncoder(SequenceEncoder): 

    def __init__(self, input_feature_dim, output_feature_dim):
        super().__init__()
        # self.cfg = cfg
        # input_feature_dim = self.cfg.get('input_feature_dim', None) or self.cfg.feature_dim 
        # output_feature_dim = self.cfg.feature_dim
        self.linear = torch.nn.Linear(input_feature_dim, output_feature_dim)

    def forward(self, sample, input_key="fused_feature"):
        feat = sample[input_key] 
        # B, T, D -> B * T, D 
        out = feat.view(-1, feat.shape[-1])
        out = self.linear(feat) 
        # B * T, D -> B, T, D
        out = out.view(feat.shape[0], feat.shape[1], -1)
        sample["seq_encoder_output"] = out 
        return sample

    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim

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

def create_squasher(input_dim, hidden_dim, quant_factor):
    layers = [nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,5,stride=2,padding=2,
                        padding_mode='replicate'),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(hidden_dim))]
    for _ in range(1, quant_factor):
        layers += [nn.Sequential(
                    nn.Conv1d(hidden_dim,hidden_dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(0.2, True),
                    nn.BatchNorm1d(hidden_dim),
                    nn.MaxPool1d(2)
                    )]
    squasher = nn.Sequential(*layers)
    return squasher