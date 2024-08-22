"""
Author: Radek Danecek
Copyright (c) 2023, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emote@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""

import torch 
import math 


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   
    else:                                                 
        closest_power_of_2 = 2**math.floor(math.log2(n)) 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_alibi_biased_mask(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the ALiBi paper.
    This means that the upper triangle of the mask is filled with -inf, 
    the diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    That lowers the attention to the past (the number gets lower the further away from the diagonal it is).
    """
    period = 1
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


def init_alibi_biased_mask_future(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the ALiBi paper but 
    with not with the future masked out.
    The diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    The upper triangle is filled symmetrically with the lower triangle.
    That lowers the attention to the past and the future (the number gets lower the further away from the diagonal it is).
    """
    period = 1
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = alibi - torch.flip(alibi, [1, 2])
    mask = alibi + torch.flip(alibi, [1, 2])
    return mask


def init_faceformer_biased_mask(num_heads, max_seq_len, period):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the FaceFormer paper.
    This means that the upper triangle of the mask is filled with -inf, 
    the diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    That lowers the attention to the past (the number gets lower the further away from the diagonal it is).
    The biased mask has a period, if larger than 1, the bias is repeated period times before coming to the next value.
    If the period is 1, the mask is the same as the alibi mask.
    """
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

def init_faceformer_biased_mask_future(num_heads, max_seq_len, period):
    """
    Returns a mask for the self-attention layer. The mask is initialized according to the FaceFormer paper but 
    with not with the future masked out. The biased mask has a perioud, after which it repeats
    The diagonal is filled with 0 and the lower triangle is filled with a linear function of the position. 
    The upper triangle is filled symmetrically with the lower triangle.
    That lowers the attention to the past and the future (the number gets lower the further away from the diagonal it is).
    The biased mask has a period, if larger than 1, the bias is repeated period times before coming to the next value.
    If the period is 1, the mask is the same as the alibi mask.
    """
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    # mask = alibi - torch.flip(alibi, [1, 2])
    mask = alibi + torch.flip(alibi, [1, 2])
    return mask

def init_mask(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. 
    The upper triangle of the mask is filled with -inf, the lower triangle is filled with 0. 
    The diagonal is filled with 0.
    """
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.unsqueeze(0).repeat(num_heads,1,1)

def init_mask_future(num_heads, max_seq_len):
    """
    Returns a mask for the self-attention layer. The mask is filled with 0s.
    """
    return torch.zeros(num_heads, max_seq_len, max_seq_len)


# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    smaller_dim = min(T, S)
    for i in range(smaller_dim):
            mask[i, i] = 0
    return (mask==1).to(device=device)
