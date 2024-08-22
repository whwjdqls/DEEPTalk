import numpy as np
import torch

class Mask :
    def __init__(self, mask_path) :
        self.all_mask = np.load(mask_path,  allow_pickle=True, encoding='latin1')

    def load_mask(self, type_, mask_shape, device, only_indice=False) :
        parts = ['eye_region', 'neck', 'left_eyeball', 'right_eyeball', 'right_ear', 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region']
        mask = torch.zeros(mask_shape, dtype=torch.bool, device=device) # (bs*t,5023,3)
        specific_masks = []
        if not isinstance(type_, list) :
            type_ = [type_]
        for t in type_ :
            if t not in parts :
                raise ValueError(f'{type_} mask is not avaliable')
                return 'mask not available'
            specific_mask = self.all_mask[t]
            specific_masks.append(specific_mask)
        concatenated_mask = np.concatenate(specific_masks, axis=-1) # lip:254
        if only_indice :
            return concatenated_mask
        mask[:, concatenated_mask] = True # (bs*t,5023,3)

        return mask
    
    # get zero padded masked vertice
    def masked_vertice(self, type_, mask_shape, vertices, device) :
        specific_mask = self.load_mask(type_, mask_shape, device)
        vertices[~specific_mask] = 0
        return vertices

    # get only masked vertice
    def vertice_masked_only(self, type_, mask_shape, vertices, device) :
        specific_mask_indices = self.load_mask(type_, mask_shape, device, only_indice=True) # lip:254
        masked_vertices = vertices[:,specific_mask_indices] #lip:(bs*t,254,3)

        return masked_vertices
        
    

    
    def get_vertices(self, type_, mask_shape, vertices, device) :
        specific_mask = self.load_mask(type_, mask_shape, device)
        return vertices[specific_mask]
    