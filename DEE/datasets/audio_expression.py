
import os
import torch
import torch.utils.data as data
import numpy as np
import sys
import json
import glob
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor
import tqdm
from scipy.signal import savgol_filter
import time
import pickle
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


RAVDESS_ids = ['Actor_01', 'Actor_02', 'Actor_03', 'Actor_04', 'Actor_05', 'Actor_06', 'Actor_07', 'Actor_08', 'Actor_09', 'Actor_10', 'Actor_11', 'Actor_12', 'Actor_13', 'Actor_14', 'Actor_15', 'Actor_16', 'Actor_17', 'Actor_18', 'Actor_19', 'Actor_20', 'Actor_21', 'Actor_22', 'Actor_23', 'Actor_24']

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

EMOTION_DICT = {'neutral': 1, 'calm': 2, 'happy': 3, 'sad': 4, 'angry' :  5, 'fear': 6, 'disgusted': 7, 'surprised': 8, 'contempt' : 9}
GENDER_DICT = {'M' : 0, 'W' : 1}

# these are the ids whose expression params were obtained at the first attempt
# which had a bug. Therefore, we are not using these ids for training
celebv_bug_ids = ['sp_0000', 'sp_0001', 'sp_0002', 'sp_0049', 'sp_0051', 'sp_0052', 'sp_0053', 'sp_0054',
                  'sp_0055', 'sp_0056', 'sp_0057', 'sp_0058', 'sp_0059']

# these are the version 1 ids 
celebv_training_ids = ['sp_0000', 'sp_0001', 'sp_0002', 'sp_0004', 'sp_0005', 'sp_0006', 'sp_0007', 'sp_0008', 
                       'sp_0009', 'sp_0010', 'sp_0011', 'sp_0013', 'sp_0014', 'sp_0015', 'sp_0020', 'sp_0027',
                       'sp_0049', 'sp_0051', 'sp_0052', 'sp_0053', 'sp_0054', 'sp_0055', 'sp_0056', 'sp_0057',
                       'sp_0058' ] # 25 ids
celebv_visualize_ids = ['sp_0059']
voxceleb_training_ids = ['voxceleb2_1', 'voxceleb2_2', 'voxceleb2_3', 'voxceleb2_4', 'voxceleb2_5']
voxceleb_validation_ids = ['voxceleb2_6']


# this can be very tricky because we want to always validate
# on MEAD what ever dataset we are using and we want it to be random slice too

class AudioExpressionDataset(data.Dataset):
    def __init__(self, args, dataset='MEAD',split='train',vox_truncate=False):
        assert args.random_slice or args.full_length, 'either random_slice or full_length should be True'
        self.split = split # split is train, val, test, debug, visualize, or single actor id
        self.dataset = dataset # change to explicitly specify the dataset
        self.vox_truncate = vox_truncate
        self.fps = 25
        if args.use_30fps:
            self.fps = 30
        # data path
        self.audio_dir = os.path.join(args.audio_feature_dir ,self.dataset , 'audio_sample')
        self.expression_feature_dir = os.path.join(args.expression_feature_dir , self.dataset ,'flame_param')
        # max sequence length
        self.audio_feature_len = args.audio_feature_len
        self.expression_feature_len = args.expression_feature_len
        # list for features
        self.inputs = []
        self.labels = []
        self.full_length = args.full_length # if ture, use whole audio and expression / batch size should be 1
        self.random_slice = args.random_slice # if true, randomly slice audio and expression
        # if random_slice, we are going to initialize the dataset with arbitrary length and slice them while getting item
        # therefore, the difference between full_length and random slice whould be the fact that 
        # full_length should make sure that the length of audio and expression is shorter than positional encoding
        self.use_jaw = True if args.affectnet_model_path is not None else False
        # when trainig wiht celebv but validating with MEAD, we are going to use random slice
        if dataset == 'MEAD' and args.dataset == 'CELEBV' and args.split == 'val':
            self.random_slice = True
        
        self.disable_padding = args.disable_padding # if true, pad the audio and expression to feature_len -> recommend to always use this
        # -> not padding leads to almost halving the dataset size when the length is set to 2 sec
        
        num_padded_audio = 0
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
            
            audio_clip =  args.clip_length * 1600
            expression_clip = args.clip_length * 3  
            audio_stride = args.stride_length * 1600 # for training, stride is 0.1 seconds
            expression_stride = args.stride_length * 3
            
            if split in ['val', 'test', 'visualize'] + training_ids + val_ids + test_ids: # Don't clip for validation or test
                # this is to make all evaluation, visualization to be consistent
                audio_stride = 1600 * 2
                expression_stride = 3 * 2
                audio_clip = 0
                expression_clip = 0
                
                if args.add_n_mog >= 1: # if we are using mog, validation is too long, shorthen validation
                    audio_stride = 1600 * 10 # always note that the validation for MOG is different from the original validation!!
                    expression_stride = 3 * 10
                    audio_clip = 1600 * 5
                    expression_clip = 3 * 5
                    
                
            file_paths = []
            for actor in actor_list:
                file_paths += glob.glob(os.path.join(self.expression_feature_dir, actor, '*.npy'))
            for file_path in tqdm.tqdm(file_paths):
                uid = file_path.split('/')[-1].split('.')[0]
                
                actor_name = uid.split('_')[0] # M005
                actor_id = MEAD_ACTOR_DICT[actor_name] # name -> id
                emotion = int(uid.split('_')[1])    
                intensity = int(uid.split("_")[2])
                gender = GENDER_DICT[uid.split('_')[0][0]] # M -> 0, W -> 1
                
                param_dict = {}
                parameters = np.load(file_path)
                try :
                    if parameters.shape[1] == 153:
                        param_dict['expression'] = parameters[:,:50] # (T, 50)
                        param_dict['jaw'] = parameters[:,50:53] # (T, 3)
                        param_dict['shape'] = parameters[:,53:]
                    elif parameters.shape[1] == 156:
                        param_dict['expression'] = parameters[:,:50]
                        param_dict['jaw'] = parameters[:,53:56]
                        param_dict['shape'] = parameters[:,56:]
                except :
                    print(f'Something wrong with {file_path}')
                    print(f'expression shape : {parameters.shape}')
                    continue
                if args.normalize_exp: # normalizing expression with respect to the temperal axis
                    param_dict['expression'] = (param_dict['expression'] - np.mean(param_dict['expression'], axis=0)) / np.std(param_dict['expression'], axis=0)
                
                expression_feature = torch.tensor(param_dict['expression'], dtype=torch.float32) #(len, 50)
                if self.use_jaw:
                    expression_feature = torch.tensor(np.concatenate([param_dict['expression'], param_dict['jaw']], axis=1), dtype=torch.float32)
                    
                if args.smooth_expression:
                    expression_feature = savgosavgol_filterl_filter(expression_feature, 5, 2, axis=0)

                audio_path = os.path.join(self.audio_dir, actor_name , uid + '.npy')
                
                if not os.path.exists(audio_path) :
                    print(f'{audio_path} doesnt exist')
                    continue
                
                audio_samples = np.load(audio_path)
                audio_samples = torch.tensor(audio_samples, dtype=torch.float32)
                
                if not self.disable_padding:
                    # pad audio and expression to feature length
                    if self.audio_feature_len > len(audio_samples):
                        audio_samples = torch.nn.functional.pad(audio_samples, (0, self.audio_feature_len - len(audio_samples)))
                        num_padded_audio += 1
                    if self.expression_feature_len > len(expression_feature): # pad (featdim front=0, featdim back=0, tempdim front=0, tempdim back=residual length)
                        expression_feature = torch.nn.functional.pad(expression_feature, (0,0,0, self.expression_feature_len - len(expression_feature)))
                        num_padded_exp += 1
                    
                # if we are using full_length or random_slice, we are going to initialize them with each full length
                if self.full_length or self.random_slice:
                    if self.full_length and ((len(audio_samples) >= args.max_seq_len * 16000) or (len(expression_feature) >= args.max_seq_len * self.fps)):
                        print(f'audio or expression is too long {len(audio_samples)} {len(expression_feature)}')
                        print(f'audio path : {audio_path}') # if using full_length, the length should not exceed the positional encoding or alibi scheme
                        continue
                    self.inputs.append([audio_samples, expression_feature])
                    self.labels.append([emotion, intensity, gender, actor_id])
                    continue
                
                for audio_start, expression_start in zip(
                    range(audio_clip, audio_samples.shape[0] - self.audio_feature_len - audio_clip, audio_stride),
                    range(expression_clip, expression_feature.shape[0] - self.expression_feature_len - expression_clip, expression_stride)
                ):
                    audio_samples_slice = audio_samples[audio_start:audio_start+self.audio_feature_len]
                    
                    expression_samples_slice = expression_feature[expression_start:expression_start+self.expression_feature_len]
                    
                    if audio_samples_slice.shape[0] != self.audio_feature_len or expression_samples_slice.shape[0] != self.expression_feature_len:
                        print(f'skipping this slice as the lengths are not aligned {len(audio_samples_slice)} {len(expression_samples_slice)}')
                        continue

                    self.inputs.append([audio_samples_slice, expression_samples_slice])
                    self.labels.append([emotion, intensity, gender, actor_id])
                    # [int, int, int, int]
                
        elif self.dataset == 'RAVDESS':
            actor_list = []
            if self.split == 'visualize': # for now, lets visualize all the actors
                actor_list = RAVDESS_ids
            elif self.split == 'debug': 
                actor_list = ['Actor_01']
            elif self.split in RAVDESS_ids: # this is for single actor
                actor_list = [self.split]
            else:
                raise NotImplementedError('split should be debug or visualize')
            print(f'making dataset with {len(actor_list)} actors {actor_list}')
            all_actor_list = os.listdir(self.audio_dir)
            
            # check if all the actors are in the directory
            if set(actor_list).intersection(set(all_actor_list)) != set(actor_list):
                print(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
                print(f'missing actors : {set(actor_list).difference(set(all_actor_list))}')
                print(f'using only {set(actor_list).intersection(set(all_actor_list))}')
                actor_list = set(actor_list).intersection(set(all_actor_list))
                raise ValueError(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
            
            audio_clip =  args.clip_length * 1600
            expression_clip = args.clip_length * 3  
            audio_stride = args.stride_length * 1600 # for training, stride is 0.1 seconds
            expression_stride = args.stride_length * 3
            
            if split in ['val', 'test', 'visualize'] + RAVDESS_ids: # Don't clip for validation or test
                # this is to make all evaluation, visualization to be consistent
                audio_stride = 1600 * 2
                expression_stride = 3 * 2
                audio_clip = 0
                expression_clip = 0   
            file_paths = []
            for actor in actor_list:
                file_paths += glob.glob(os.path.join(self.expression_feature_dir, actor, '*.npy'))
                # (NOTE) file_paths is extracted from expression directory which we have made 
                # therefore, the filenames are in the order fo Actor_EMotion_Intensity_Statement_Repetition.npy
            for file_path in tqdm.tqdm(file_paths):
                uid = file_path.split('/')[-1].split('.')[0]
                
                actor_name = 'Actor_' + uid.split('-')[-1]
                actor_id = int(uid.split('-')[-1])
                emotion = int(uid.split('-')[2])
                intensity = int(uid.split('-')[3])
                statement = int(uid.split('-')[4]) # 1 -> "Kids are talking by the door", 2 -> "Dogs are sitting by the door"
                repetition = int(uid.split('-')[5]) # 1 -> 1st repetition, 2 -> 2nd repetition
                if actor_id % 2 == 0: # if actor num is even
                    gender = 1 # actor is female
                else: # if actor num is odd
                    gender = 0 # actor is male
                
                # continue if a specific statement or repetition is specified
                if args.statement is not None and statement != args.statement:
                    continue
                if args.repetition is not None and repetition != args.repetition:
                    continue
                
                param_dict = {}
                parameters = np.load(file_path)
                try :
                    if parameters.shape[1] == 153:
                        param_dict['expression'] = parameters[:,:50] # (T, 50)
                        param_dict['jaw'] = parameters[:,50:53] # (T, 3)
                        param_dict['shape'] = parameters[:,53:]
                    elif parameters.shape[1] == 156:
                        param_dict['expression'] = parameters[:,:50]
                        param_dict['jaw'] = parameters[:,53:56]
                        param_dict['shape'] = parameters[:,56:]
                except :
                    print(f'Something wrong with {file_path}')
                    print(f'expression shape : {parameters.shape}')
                    continue
                if args.normalize_exp: # normalizing expression with respect to the temperal axis
                    param_dict['expression'] = (param_dict['expression'] - np.mean(param_dict['expression'], axis=0)) / np.std(param_dict['expression'], axis=0)
                    
                expression_feature = torch.tensor(param_dict['expression'], dtype=torch.float32) #(len, 50)
                if self.use_jaw:
                    expression_feature = torch.tensor(np.concatenate([param_dict['expression'], param_dict['jaw']], axis=1), dtype=torch.float32)
                    
                if args.smooth_expression:
                    expression_feature = savgol_filter(expression_feature, 5, 2, axis=0)

                audio_path = os.path.join(self.audio_dir, actor_name , uid + '.npy')
                audio_samples = np.load(audio_path)


                audio_samples = torch.tensor(audio_samples, dtype=torch.float32)
                
                if not self.disable_padding:
                    # pad audio and expression to feature length
                    if self.audio_feature_len > len(audio_samples):
                        audio_samples = torch.nn.functional.pad(audio_samples, (0, self.audio_feature_len - len(audio_samples)))
                        num_padded_audio += 1
                    if self.expression_feature_len > len(expression_feature): # pad (featdim front=0, featdim back=0, tempdim front=0, tempdim back=residual length)
                        expression_feature = torch.nn.functional.pad(expression_feature, (0,0,0, self.expression_feature_len - len(expression_feature)))
                        num_padded_exp += 1
                    
                if self.full_length or self.random_slice:
                    if self.full_length and (len(audio_samples) >= args.max_seq_len * 16000) or (len(expression_feature) >= args.max_seq_len * self.fps):
                        print(f'audio or expression is too long {len(audio_samples)} {len(expression_feature)}')
                        print(f'audio path : {audio_path}') # if using full_length, the length should not exceed the positional encoding or alibi scheme
                        continue
                    
                    # full length audio
                    self.inputs.append([audio_samples, expression_feature])
                    self.labels.append([emotion, intensity, gender, actor_id])
                    continue
                
                for audio_start, expression_start in zip(
                    range(audio_clip, audio_samples.shape[0] - self.audio_feature_len - audio_clip, audio_stride),
                    range(expression_clip, expression_feature.shape[0] - self.expression_feature_len - expression_clip, expression_stride)
                ):
                    audio_samples_slice = audio_samples[audio_start:audio_start+self.audio_feature_len]
                    
                    expression_samples_slice = expression_feature[expression_start:expression_start+self.expression_feature_len]
                    
                    if audio_samples_slice.shape[0] != self.audio_feature_len or expression_samples_slice.shape[0] != self.expression_feature_len:
                        continue
                     
                    self.inputs.append([audio_samples_slice, expression_samples_slice])
                    self.labels.append([emotion, intensity, gender, actor_id])
                    
        elif self.dataset == 'CELEBV':
            discarded_num = 0
            if self.full_length:
                raise NotImplementedError('CELEBV does not support full_length')
            actor_list = []
            if self.split == 'train':
                actor_list = celebv_training_ids
            elif self.split == 'debug': 
                actor_list = ['sp_0004']
            elif self.split == 'visualize':
                actor_list = celebv_visualize_ids
            elif self.split == 'val':
                raise NotImplementedError('val is not implemented for CELEBV as there is no emotion label')
            else:
                raise NotImplementedError('split should be train, or debug')
            
            print(f'as there is a bug in the dataset, we are not using {celebv_bug_ids}!!!')
            actor_list_ = actor_list.copy()
            for actor in actor_list_:
                if actor in celebv_bug_ids:
                    actor_list.remove(actor)
                    print(f'removing {actor} from the actor list as it has a bug')
            
            print(f'making dataset with {len(actor_list)} actors {actor_list}')
            all_actor_list = os.listdir(self.expression_feature_dir)
            
            # check if all the actors are in the directory
            if set(actor_list).intersection(set(all_actor_list)) != set(actor_list):
                print(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
                print(f'missing actors : {set(actor_list).difference(set(all_actor_list))}')
                print(f'using only {set(actor_list).intersection(set(all_actor_list))}')
                actor_list = set(actor_list).intersection(set(all_actor_list))
                # raise ValueError(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
                
            # for celebv, we recommend using stride size same as the feature length
            audio_clip =  args.clip_length * 1600
            expression_clip = args.clip_length * 3  
            audio_stride = args.stride_length * 1600 
            expression_stride = args.stride_length * 3
            
                
            file_paths = []
            for actor in actor_list:
                file_paths += glob.glob(os.path.join(self.expression_feature_dir, actor, '*.npy'))
                
            for file_path in tqdm.tqdm(file_paths):
                uid = file_path.split('/')[-1].split('.')[0]
                
                actor_name = file_path.split('/')[-2] # sp_0000
                # dummy values for emotion, intensity and gender for CELEBV
                emotion = 0
                intensity = 0
                gender = 0
                actor_id = 0
                
                param_dict = {}
                parameters = np.load(file_path)
                try :
                    param_dict['expression'] = parameters[:,:50]
                    param_dict['jaw'] = parameters[:,50:53]
                    param_dict['shape'] = parameters[:,53:]
                except :
                    print(f'Something wrong with {file_path}')
                    print(f'expression shape : {parameters.shape}')
                    continue
                
                if args.normalize_exp: # normalizing expression with respect to the temperal axis
                    param_dict['expression'] = (param_dict['expression'] - np.mean(param_dict['expression'], axis=0)) / np.std(param_dict['expression'], axis=0)
                    
                expression_feature = torch.tensor(param_dict['expression'], dtype=torch.float32) #(len, 50)
                if self.use_jaw:
                    expression_feature = torch.tensor(np.concatenate([param_dict['expression'], param_dict['jaw']], axis=1), dtype=torch.float32)
                    
                if args.smooth_expression:
                    expression_feature = savgol_filter(expression_feature, 5, 2, axis=0)

                audio_path = os.path.join(self.audio_dir, actor_name , uid + '.npy')

                if not os.path.exists(audio_path) :
                    print(f'{audio_path} doesnt exist')
                    continue
                
                audio_samples = np.load(audio_path)
                audio_samples = torch.tensor(audio_samples, dtype=torch.float32)
                # for CElebc, we do not use padding, we just discard the samples that are shorter than the feature length
                
                if self.audio_feature_len > len(audio_samples):
                    discarded_num += 1
                    print(f'{audio_path} is shorter than the feature length {len(audio_samples)}')
                    continue
                if self.expression_feature_len > len(expression_feature):
                    discarded_num += 1  
                    print(f'{file_path} is shorter than the feature length {len(expression_feature)}')
                    continue

                # if we are using full_length or random_slice, we are going to initialize them with each full length
                if self.random_slice:
                    self.inputs.append([audio_samples, expression_feature])
                    self.labels.append([emotion, intensity, gender, actor_id])
                    continue
                
                for audio_start, expression_start in zip(
                    range(audio_clip, audio_samples.shape[0] - self.audio_feature_len - audio_clip, audio_stride),
                    range(expression_clip, expression_feature.shape[0] - self.expression_feature_len - expression_clip, expression_stride)
                ):
                    audio_samples_slice = audio_samples[audio_start:audio_start+self.audio_feature_len]
                    
                    expression_samples_slice = expression_feature[expression_start:expression_start+self.expression_feature_len]
                    
                    if audio_samples_slice.shape[0] != self.audio_feature_len or expression_samples_slice.shape[0] != self.expression_feature_len:
                        print(f'skipping this slice as the lengths are not aligned {len(audio_samples_slice)} {len(expression_samples_slice)}')
                        continue
                
                    self.inputs.append([audio_samples_slice, expression_samples_slice])
                    self.labels.append([emotion, intensity, gender, actor_id])
                    
            # discarding shot samples in celebv
            print(f'{discarded_num} samples are discarded as they are shorter than the feature length')
        elif self.dataset == 'Voxceleb':  
            actor_list = []
            if self.split == 'train':
                actor_list = voxceleb_training_ids
            elif self.split == 'val':
                actor_list = voxceleb_validation_ids
            elif self.split == 'debug':
                actor_list = voxceleb_validation_ids
            elif self.split == 'visualize':
                actor_list = voxceleb_validation_ids
            elif self.split in voxceleb_training_ids + voxceleb_validation_ids : # for single actor
                actor_list = [self.split]
            else:
                raise NotImplementedError('split should be train, val, test, debug, visualize or single actor id')
                

            audio_clip =  args.clip_length * 1600
            expression_clip = int(args.clip_length * self.fps / 10)
            audio_stride = args.stride_length * 1600 # for training, stride is 0.1 seconds
            expression_stride = int(args.stride_length * self.fps / 10)
            
            if split in ['val', 'test', 'visualize'] + training_ids + val_ids + test_ids: # Don't clip for validation or test
                # this is to make all evaluation, visualization to be consistent
                audio_stride = 1600 * 2
                # expression_stride = 3 * 2
                expression_stride = int(self.fps /10 *2)
                audio_clip = 0
                expression_clip = 0

                
            file_paths = []
            for actor in actor_list :
                with open(f'{args.voxceleb_pickle}/{actor}_filtered_path.pkl', 'rb') as f :
                    pickle_path = pickle.load(f)
                file_paths += pickle_path

            for file_path in tqdm.tqdm(file_paths):
                path_split = file_path.split('/')
                audio_name = f'{path_split[-3]}/{path_split[-2]}/{path_split[-1]}'
                actor_name = path_split[-5]
                param_dict = {}
                parameters = np.load(file_path)
                try :
                    param_dict['expression'] = parameters[:,:50] # (T, 50)
                    param_dict['jaw'] = parameters[:,53:56]
                    param_dict['shape'] = parameters[:,56:]
                except :
                    print(f'Something wrong with {file_path}')
                    print(f'expression shape : {parameters.shape}')
                    continue
                if args.normalize_exp: # normalizing expression with respect to the temperal axis
                    param_dict['expression'] = (param_dict['expression'] - np.mean(param_dict['expression'], axis=0)) / np.std(param_dict['expression'], axis=0)

                expression_feature = torch.tensor(np.concatenate([param_dict['expression'], param_dict['jaw']], axis=1), dtype=torch.float32)
                    
                if args.smooth_expression:
                    expression_feature = savgol_filter(expression_feature, 5, 2, axis=0)

                audio_path = os.path.join(args.audio_feature_dir, actor_name, 'audio_sample', audio_name)
                
                if not os.path.exists(audio_path) :
                    print(f'{audio_path} doesnt exist')
                    continue
                
                try :
                    audio_samples = np.load(audio_path)
                    audio_samples = torch.tensor(audio_samples, dtype=torch.float32)
                except :
                    print(f'Something wrong with {audio_path}')
                    continue
                
                if not self.disable_padding:
                    # pad audio and expression to feature length
                    if self.audio_feature_len > len(audio_samples):
                        print(f'pad audio_sample : {len(audio_samples)} / expression : {len(expression_feature)}')
                        audio_samples = torch.nn.functional.pad(audio_samples, (0, self.audio_feature_len - len(audio_samples)))
                        num_padded_audio += 1
                    if self.expression_feature_len > len(expression_feature): # pad (featdim front=0, featdim back=0, tempdim front=0, tempdim back=residual length)
                        print(f'pad expression_sample : {len(expression_feature)} / audio : {len(audio_samples)}')
                        expression_feature = torch.nn.functional.pad(expression_feature, (0,0,0, self.expression_feature_len - len(expression_feature)))
                        num_padded_exp += 1
                    
                # if we are using full_length or random_slice, we are going to initialize them with each full length
                if self.full_length or self.random_slice:
                    if self.full_length and ((len(audio_samples) >= args.max_seq_len * 16000) or (len(expression_feature) >= args.max_seq_len * self.fps)):
                        if self.vox_truncate:
                            audio_samples = audio_samples[:args.max_seq_len * 16000]
                            expression_feature = expression_feature[:args.max_seq_len * self.fps]
                            print(f'truncated audio and expression to {args.max_seq_len} seconds')
                        else:
                            print(f'audio or expression is too long {len(audio_samples)} {len(expression_feature)}')
                            print(f'audio path : {audio_path}') # if using full_length, the length should not exceed the positional encoding or alibi scheme
                            continue
                    self.inputs.append([audio_samples, expression_feature])
                    # self.labels.append([emotion, intensity, gender, actor_id])
                    self.labels.append(file_path)
                    continue
                
                for audio_start, expression_start in zip(
                    range(audio_clip, audio_samples.shape[0] - self.audio_feature_len - audio_clip, audio_stride),
                    range(expression_clip, expression_feature.shape[0] - self.expression_feature_len - expression_clip, expression_stride)
                ):
                    audio_samples_slice = audio_samples[audio_start:audio_start+self.audio_feature_len]
                    
                    expression_samples_slice = expression_feature[expression_start:expression_start+self.expression_feature_len]
                    
                    if audio_samples_slice.shape[0] != self.audio_feature_len or expression_samples_slice.shape[0] != self.expression_feature_len:
                        print(f'skipping this slice as the lengths are not aligned {len(audio_samples_slice)} {len(expression_samples_slice)}')
                        continue
        
                    self.inputs.append([audio_samples_slice, expression_samples_slice])
                    self.labels.append(file_path)
                    # self.labels.append([emotion, intensity, gender, actor_id])
                    # [int, int, int, int]

        print(f'{self.split} dataset loading finished!!')
        print(f'num_padded_exp: {num_padded_exp}')
        print(f'num_padded_audio: {num_padded_audio}')
        print(f'num_inputs : {len(self.inputs)}')
    
    def __getitem__(self, index):
        if self.random_slice: # please optimize this shitty code later..
            audio, expression = self.inputs[index] # arbitrary length
            
            flag = True
            while(flag):
                if (len(audio) - self.audio_feature_len) == 0:
                    audio_start = 0
                else:
                    audio_start = np.random.randint(0, len(audio) - self.audio_feature_len)
                    
                exp_start = int(audio_start * self.fps / 16000)
                audio_ = audio[audio_start:audio_start+self.audio_feature_len]
                expression_ = expression[exp_start:exp_start+self.expression_feature_len]
                if len(audio_) == self.audio_feature_len and len(expression_) == self.expression_feature_len:
                    flag = False
                    audio = audio_
                    expression = expression_
            return [audio, expression], self.labels[index]
        return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return len(self.inputs)

class AudioDataset(data.Dataset):
    def __init__(self, args, dataset='MEAD', split='train'):
        raise NotImplementedError('AudioDataset is not implemented yet')
        self.split = split
        self.dataset = dataset
        self.audio_dir = os.path.join(args.audio_feature_dir ,self.dataset , 'audio_sample')

        # list for features
        self.inputs = []
        self.labels = []
        
        if self.MEAD:
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
            else:
                raise NotImplementedError('split should be train, val, test, debug or visualize')
            print(f'making dataset with {len(actor_list)} actors {actor_list}')
            all_actor_list = os.listdir(self.audio_dir)
            
            # check if all the actors are in the directory
            if set(actor_list).intersection(set(all_actor_list)) != set(actor_list):
                print(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
                print(f'missing actors : {set(actor_list).difference(set(all_actor_list))}')
                print(f'using only {set(actor_list).intersection(set(all_actor_list))}')
                actor_list = set(actor_list).intersection(set(all_actor_list))
                raise ValueError(f'actor_list {actor_list} is not a subset of all_actor_list {all_actor_list}')
                
            file_paths = []
            for actor in actor_list:
                file_paths += glob.glob(os.path.join(self.audio_dir, actor, '*.npy'))
            
            for file_path in tqdm.tqdm(file_paths):
                uid = file_path.split('/')[-1].split('.')[0]
                
                actor_name = uid.split('_')[0] # M005
                actor_id = MEAD_ACTOR_DICT[actor_name] # name -> id
                emotion = int(uid.split('_')[1])    
                intensity = int(uid.split("_")[2])
                gender = GENDER_DICT[uid.split('_')[0][0]] # M -> 0, W -> 1
                
                audio_path = os.path.join(self.audio_dir, actor_name , uid + '.npy')

                if not os.path.exists(audio_path) :
                    print(f'{audio_path} doesnt exist')
                    continue
                
                audio_samples = np.load(audio_path)
                audio_samples = torch.tensor(audio_samples, dtype=torch.float32)
                
                self.inputs.append(audio_samples)
                self.labels.append([emotion, intensity, gender, actor_id])
                    # [int, int, int, int]
        elif self.RAVDESS:

            raise NotImplementedError('RAVDESS is not implemented yet')
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return len(self.inputs)
    

        