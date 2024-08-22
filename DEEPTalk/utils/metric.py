import pickle
import torch
from .mask import Mask

def cal_std(masked_vertice) :
    L2_dis_upper = torch.transpose(masked_vertice,0,1)
    L2_dis_upper = torch.sum(L2_dis_upper, dim=2)
    motion_std = torch.std(L2_dis_upper, dim=0)
    motion_std_mean = torch.mean(motion_std)
    # print(f'mean shape : {motion_std.shape}')

    # return motion_std.item()
    return motion_std_mean

def FDD(template_vertices, vertices_pred, vertices_gt, mask_model, device) :
    upper_type = ['eye_region', 'forehead', 'nose', ]
    # calc motion
    motion_pred = vertices_pred - template_vertices.unsqueeze(0)
    motion_gt = vertices_gt - template_vertices.unsqueeze(0)
    # masking only upper part
    masked_motion_pred = mask_model.masked_vertice(upper_type, motion_pred.shape, motion_pred, device)
    masked_motion_gt = mask_model.masked_vertice(upper_type, motion_gt.shape, motion_gt, device)
    # calc motion std
    pred_motion_std = cal_std(masked_motion_pred)
    print(f'pred : {pred_motion_std}')
    gt_motion_std = cal_std(masked_motion_gt)
    print(f'gt : {gt_motion_std}')
    # calc FDD score
    motion_std_diff = gt_motion_std - pred_motion_std
    FDD_score = abs(motion_std_diff)
    # FDD_score = torch.sum(motion_std_diff, dim=0)/motion_std_diff.shape[0]
    
    return FDD_score


