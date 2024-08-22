import os
import glob
import torch
import json
import numpy as np
import torch.utils.data as data
import tqdm
from scipy.signal import savgol_filter
# from .dataset_utils import get_FLAME_params_RAVDESS, get_FLAME_params_MEAD
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor
import librosa
import time

# from .dataset_utils import get_FLAME_params_MEAD
RAVDESS_ACTOR_DICT = {1 : 0, 3 : 1, 4 : 2, 5 : 3, 6 : 4, 7 : 5, 8 : 6, 9 : 7, 10 : 8, 11 : 9, 12 : 10, 13 : 11, 14 : 12, 15 : 13, 16 : 14, 17 : 15, 18 : 16, 19 : 17, 20 : 18, 21 : 19, 22 : 20, 23 : 21, 24 : 22, 25 : 23, # for train
                      2 : 24} # for val


training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029'
                ] # 32 ids
val_ids = ['M032', 'M033', 'M034', 'M035', 'W033', 'W035', 'W036']  # 7 ids

test_ids = ['M037', 'M039', 'M040', 'M041', 'M042', 'W037', 'W038', 'W040'] # 8 ids

                    # 32 train_ids
MEAD_ACTOR_DICT = {'M003': 0, 'M005': 1, 'M007': 2, 'M009': 3, 'M011': 4, 'M012': 5, 'M013': 6, 'M019': 7, 'M022': 8, 'M023': 9, 'M024': 10, 'M025': 11, 'M026': 12, 'M027': 13, 'M028': 14, 'M029': 15, 'M030': 16, 'M031': 17, 'W009': 18, 'W011': 19, 'W014': 20, 'W015': 21, 'W016': 22, 'W018': 23, 'W019': 24, 'W021': 25, 'W023': 26, 'W024': 27, 'W025': 28, 'W026': 29, 'W028': 30, 'W029': 31, 
                   'M032': 32, 'M033': 33, 'M034': 34, 'M035': 35, 'W033': 36, 'W035': 37, 'W036': 38, # 7 val_ids
                   'M037': 39, 'M039': 40, 'M040': 41, 'M041': 42, 'M042': 43, 'W037': 44, 'W038': 45, 'W040': 46} # 8 test_ids

# EMOTION_DICT = {'neutral': 1, 'calm': 2, 'happy': 3, 'sad': 4, 'angry' :  5, 'fear': 6, 'disgusted': 7, 'surprised': 8, 'contempt' : 9}
# calm for RAVDESS
EMOTION_DICT = {'neutral': 1, 'happy': 2, 'sad': 3, 'surprised': 4, 'fear': 5, 'disgusted': 6, 'angry': 7, 'contempt': 8, 'calm' : 9}
# modify DICT to match inferno's original emotion label
modify_DICT = {1:1, 3:2, 4:3, 5:7, 6:5, 7:6, 8:4, 9:8}
GENDER_DICT = {'M' : 0, 'W' : 1}

def pad_exp_to_match_quantfactor(flame_params, fps=25, quant_factor=3) :
    """padidng audio samples to be divisible by quant factor
    (NOTE) quant factor means the latents must be divisible by 2^(quant_factor)
           for inferno's EMOTE checkpoint, the quant_factor is 3 and fps is 25
    Args:
        audio_samples (torch tensor or numpy array): audio samples from raw wav files 
        fps (int, optional): fps of the face parameters. Defaults to 30.
        quant_factor (int, optional): squaushing latent variables by 2^(quant_factor) Defaults to 8.
    """
    if isinstance(flame_params, np.ndarray):
        flame_params = torch.tensor(flame_params, dtype=torch.float32)
    
    latent_len = flame_params.shape[0]
    target_len = latent_len + (2**quant_factor - (latent_len % (2**quant_factor) )) # make sure the length is divisible by quant factor
    padded_flame_params = torch.nn.functional.pad(flame_params, (0,0,0, target_len - len(flame_params)))
    
    if isinstance(flame_params, np.ndarray):
        padded_flame_params = padded_flame_params.numpy()
        
    return padded_flame_params

class MotionPriorMEADDataset(data.Dataset):
    def __init__(self, config, split='train'):
        # NOTE : config should be config['data']['train'] or config['data']['val']
        # split
        self.split = split
        self.config = config
        self.dataset = config['dataset']
        # data path
        self.expression_feature_dir = config["expression_dir"]
        self.smooth_expression = config["smooth_expression"]
        # for other configs, window_size maybe in seconds
        # but for motion prior dataset, window_size is in frames!!
        self.window_size = config["window_size"] # T in EMOTE paper
        self.start_clip = config["start_clip"]
        self.end_clip = config["end_clip"]
        self.stride = config["stride"]
        # list for features
        self.inputs = []
        
        self.full_length = config["full_length"] # use full length of the data 
        # for dataloader, the batch size should be 1
        self.random_slice = config["random_slice"] # random slice from each clip
        if self.full_length and self.random_slice:
            raise ValueError('full_length and random_slice cannot be True at the same time')    
        num_padded_exp = 0 
        
        if self.dataset == 'MEAD':
            actor_list = []
            if self.split == 'train':
                actor_list = training_ids
            elif self.split == 'val':
                actor_list = val_ids
            elif self.split == 'test':
                actor_list = test_ids
            elif self.split == 'debug':
                actor_list = ['M003']
            elif self.split == 'visualize':
                actor_list = ['M032', 'M033']
            elif self.split in training_ids + val_ids + test_ids: # for single actor
                actor_list = [self.split]
            else:
                raise NotImplementedError('split should be train, val, test, debug, visualize or single actor id')
            print(f'making dataset with {len(actor_list)} actors {actor_list}')
            all_actor_list = os.listdir(self.expression_feature_dir)
            # check if all the actors are in the directory
            if set(actor_list).intersection(set(all_actor_list)) != set(actor_list):
                print(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
                print(f'missing actors : {set(actor_list).difference(set(all_actor_list))}')
                print(f'using only {set(actor_list).intersection(set(all_actor_list))}')
                actor_list = set(actor_list).intersection(set(all_actor_list))
                raise ValueError(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
            

            exp_start, exp_end, exp_stride, exp_window = np.array([self.start_clip, self.end_clip, self.stride, self.window_size])
            file_paths = []
            for actor in actor_list:
                file_paths += glob.glob(os.path.join(self.expression_feature_dir, actor, '*.npy'))
            
            for file_path in tqdm.tqdm(file_paths):
                param_dict = {}
                parameters = np.load(file_path)
                try :
                    param_dict['expression'] = parameters[:,:50]
                    # param_dict['jaw'] = parameters[:,50:53]
                    # param_dict['shape'] = parameters[:,53:]
                    param_dict['jaw'] = parameters[:,53:56]
                    param_dict['shape'] = parameters[:,56:]
                except :
                    print(f'Something wrong with {file_path}')
                    continue

                expression_feature = torch.tensor(param_dict['expression'], dtype=torch.float32) #(len, 50)
                jaw_feature = torch.tensor(param_dict['jaw'], dtype=torch.float32) #(len,3)
                if exp_window > len(expression_feature): # pad (featdim front=0, featdim back=0, tempdim front=0, tempdim back=residual length)
                    expression_feature = torch.nn.functional.pad(expression_feature, (0,0,0, exp_window - len(expression_feature)))
                    jaw_feature = torch.nn.functional.pad(jaw_feature, (0,0,0, exp_window - len(jaw_feature)))
                    num_padded_exp += 1
                param_feature = torch.cat([expression_feature, jaw_feature], dim=1)
                
                if self.smooth_expression:
                    param_feature = savgol_filter(param_feature, 5, 2, axis=0)
                if self.random_slice or self.full_length:
                    if self.full_length:
                        param_feature = pad_exp_to_match_quantfactor(param_feature, fps=25, quant_factor=3) # fps : 25
                    self.inputs.append(param_feature)
                    continue
                for expression_start_ in range(exp_start, param_feature.shape[0] - exp_window - exp_end, exp_stride):
                    expression_samples_slice = param_feature[expression_start_:expression_start_+exp_window]
                    if expression_samples_slice.shape[0] != exp_window:
                        continue

                    self.inputs.append(expression_samples_slice)
            print(f'Total padded dataset : {num_padded_exp}')
        else:
            raise NotImplementedError('dataset should be MEAD')

    def __getitem__(self,index):
        if self.random_slice: # please optimize this shitty code later..
            param = self.inputs[index] # arbitrary length
            flag = True
            while(flag):
                if (len(param) - self.window_size) == 0:
                    param_start = 0
                else:
                    param_start = np.random.randint(0, len(param) - self.window_size)
                    
                param_ = param[param_start:param_start+self.window_size]
                if len(param_) == self.window_size:
                    flag = False
                    param = param_
            return param
        return self.inputs[index]
    
    def __len__(self):
        return len(self.inputs)