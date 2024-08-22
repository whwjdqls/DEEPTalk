import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import json
import pickle
import tqdm

class AffectNetExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_path, 
                 label_file_path, 
                 exclude_7=False,
                 exception_file_path=None,
                #  vertex_mask_path=None,
                 split='train'):
        self.data_file_path = data_file_path # path to the pkl file
        self.label_file_path = label_file_path # path to label pkl file
        
        self.split = split
        
        self.data = pickle.load(open(self.data_file_path, 'rb'))
        self.label_dict = pickle.load(open(self.label_file_path, 'rb'))
        
        self.data = list(self.data[self.split].items()) # list of tuples
        self.label_dict = self.label_dict[self.split]
        
    
        if exception_file_path is not None:
            assert self.split in ['train'], 'exception_file_path is only available for train split'
            exception_dict = pickle.load(open(exception_file_path, 'rb'))
            exception_dict = exception_dict[split]
            self.data = [data for data in self.data if data[0] not in exception_dict]
            print(f'Exception file is loaded. {len(exception_dict)} data is removed from the dataset')
        
        if exclude_7:
            self.data = [data for data in self.data if self.label_dict[data[0]]['expression'] != 7]
            print(f'7th expression is removed from the dataset.')
            print(f'{len(self.data)} data is remained')
  
        # self.vertex_mask = None
        # if vertex_mask_path is not None:
        #     with open(vertex_mask_path, 'rb') as file:
        #         self.vertex_mask = pickle.load(file, encoding='latin1')
        # self.neutral_vertex = None
        # if neutral_mesh_path is not None:
        #     self.neutral_vertex = np.load(neutral_mesh_path)
        
    def __len__(self):
        return len(self.data)
    
    def get_item_by_id(self, data_id):
        data_dict = dict(self.data)
        data = data_dict[data_id]
        label = self.label_dict[data_id]['expression']
        return data, label    

    def __getitem__(self, idx):
        data_id, data = self.data[idx]
        label = self.label_dict[data_id]['expression']
        return data, label
    
