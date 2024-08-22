import numpy as np
import torch

class Mask :
    def __init__(self, mask_path) :
        self.all_mask = np.load(mask_path,  allow_pickle=True, encoding='latin1')

    def load_mask(self, type_, mask_shape, device) :
        parts = ['eye_region', 'neck', 'left_eyeball', 'right_eyeball', 'right_ear', 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region']
        mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
        specific_masks = []
        if not isinstance(type_, list) :
            type_ = [type_]
        for t in type_ :
            if t not in parts :
                raise ValueError(f'{type_} mask is not avaliable')
                return 'mask not available'
            specific_mask = self.all_mask[t]
            specific_masks.append(specific_mask)
        concatenated_mask = np.concatenate(specific_masks, axis=-1)
        mask[:, concatenated_mask] = True

        return mask
    
    def masked_vertice(self, type_, mask_shape, vertices, device) :
        specific_mask = self.load_mask(type_, mask_shape, device)
        vertices[~specific_mask] = 0
        return vertices
    
