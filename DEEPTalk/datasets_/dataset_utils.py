
import json
import torch
import numpy as np



def get_FLAME_params_MEAD(param_path, jaw=False, shape=False):
    param_dict = {}
    with open(param_path) as f:
        parameters = json.load(f)
    
    # the json file keys are not integers + not sorted + start with 1
    # use their int value to sort
    dict_episode = {k: parameters[k] for k in parameters}
    for episode, chunks in dict_episode.items() :# episode starts with 1
        param_dict[episode] = {}
        param_dict[episode]['expression'] = []
        if jaw:
            param_dict[episode]['jaw'] = []
        if shape:
            param_dict[episode]['shape'] = []

        sorted_dict = {k: chunks[k] for k in sorted(chunks, key=lambda x: int(x))}
        for frame, params in sorted_dict.items():# frame starts with 1
            param_dict[episode]['expression'].append(params['expression'][:50])
            if jaw:
                param_dict[episode]['jaw'].append(params['jaw'])
            if shape:
                param_dict[episode]['shape'].append(params['shape'])
    return param_dict
# {'1' : {'expression' : [flame_param_1, flame_param_2, ...]}, '2' : }


# (10/15) new version as we concatenated all json files for each session in to one file
# this function is used for Json files containig ONE SESSION!!!
def get_FLAME_params_RAVDESS(param_path, jaw=False, shape=False):
    param_dict = {}
    param_dict['expression'] = []
    if jaw:
        param_dict['jaw'] = []
    if shape:
        param_dict['shape'] = []
    with open(param_path) as f:
        parameters = json.load(f)
    
    # the json file keys are not integers + not sorted + start with 1
    # use their int value to sort
    sorted_dict = {k: parameters[k] for k in sorted(parameters, key=lambda x: int(x))}
    for frame, params in sorted_dict.items():# frame starts with 1
        param_dict['expression'].append(params['expression'])
        if jaw:
            param_dict['jaw'].append(params['jaw'])
        if shape:
            param_dict['shape'].append(params['shape'])
    return param_dict
