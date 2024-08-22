import os, random
import torch
import numpy as np

def seed_everything(seed: int): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    # Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False causes 
    # cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    torch.backends.cudnn.benchmark = False # -> Might want to set this to True if it's too slow

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def class_from_str(str, module=None, none_on_fail = False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")

def get(cfg, key, value) :
    try :
        val = cfg[key]
    except :
        val = value
    return val

def get_checkpoint(cfg, replace_root = None, relative_to = None, checkpoint_mode=None, pattern=None):
    if checkpoint_mode is None:
        checkpoint_mode = 'latest'
        if hasattr(cfg['learning'], 'checkpoint_after_training'):
            checkpoint_mode = cfg['learning']['checkpoint_after_training']
    checkpoint = locate_checkpoint(cfg, replace_root = replace_root,
                                   relative_to = relative_to, mode=checkpoint_mode, pattern=pattern)
    return checkpoint

def get_checkpoint_with_kwargs(cfg, prefix, replace_root = None, relative_to = None, checkpoint_mode=None, pattern=None):
    checkpoint = get_checkpoint(cfg, replace_root = replace_root,
                                relative_to = relative_to, checkpoint_mode=checkpoint_mode, pattern=pattern)
    cfg['model']['resume_training'] = False  # make sure the training is not magically resumed by the old code
    # checkpoint_kwargs = {
    #     "model_params": cfg.model,
    #     "learning_params": cfg.learning,
    #     "inout_params": cfg.inout,
    #     "stage_name": prefix
    # }
    checkpoint_kwargs = {'config': cfg}
    return checkpoint, checkpoint_kwargs

# def compare_checkpoint_model(checkpoint, model):
    # """
    # Compare the model with the checkpoint
    # :param checkpoint: tcheckpoint
    # :param model: the model
    # :return: True if the model is the same as the checkpoint
    # """
    # model_dict = model.state_dict()
    # if len(checkpoint) != len(model_dict):
    #     raise RuntimeError(f'model{len(model_dict)} and checkpoint {len(checkpoint)} length are not the same')
    # for key in checkpoint:
    #     if key not in model_dict:
    #         raise RuntimeError(f'key {key} not in model')
    #     if checkpoint[key].shape != model_dict[key].shape:
    #         raise RuntimeError(f'shape of {key} in model {model_dict[key].shape} and checkpoint {checkpoint[key].shape} are not the same')
    #     if not torch.equal(checkpoint[key].to('cpu'), model_dict[key].to('cpu')):
    #         raise RuntimeError(f'values of {key} in model and checkpoint are not the same')
    #     if not torch.equal(checkpoint[key].to('cpu').dtype,  model_dict[key].to('cpu').dtype) :
    #         raise RuntimeError(f'types of {key} in model and checkpoint are not the same')
    #     # if key.startswith('sequence_decoder.motion_prior') :
    #     print(f'checkpoint : {checkpoint[key].to("cpu").dtype}')
    #     print(f'model loaded : {model_dict[key].to("cpu").dtype}')
    # return True


def compare_checkpoint_model(checkpoint, model):
    """
    Compare the model with the checkpoint
    :param checkpoint: tcheckpoint
    :param model: the model
    :return: True if the model is the same as the checkpoint
    """
    model_dict = model.state_dict()
    if len(checkpoint) != len(model_dict):
        return False
    
    for key in checkpoint:
        if key not in model_dict:
            raise ValueError(f"Checkpoint key {key} is not in model_dict")
        if checkpoint[key].shape != model_dict[key].shape:
            raise ValueError(f"Checkpoint shape {checkpoint[key].shape} is not the same as model shape {model_dict[key].shape}")
        if checkpoint[key].dtype != model_dict[key].dtype:
            raise ValueError(f"Checkpoint dtype {checkpoint[key].dtype} is not the same as model dtype {model_dict[key].dtype}")
        if not torch.equal(checkpoint[key].to('cpu'), model_dict[key].to('cpu')):
            raise ValueError(f"Checkpoint {checkpoint[key]} is not the same as model {model_dict[key]}")
        # if not torch.equal(checkpoint[key].to('cpu').dtype,  model_dict[key].to('cpu').dtype) :
        #     raise RuntimeError(f'types of {key} in model and checkpoint are not the same')

    print('_'*100)
    for key in model_dict:
        if key not in checkpoint:
            raise ValueError(f"Model key {key} is not in checkpoint")
        if checkpoint[key].to('cpu').shape != model_dict[key].to('cpu').shape:
            raise ValueError(f"Checkpoint shape {checkpoint[key].shape} is not the same as model shape {model_dict[key].shape}")
        if not torch.equal(checkpoint[key].to('cpu'), model_dict[key].to('cpu')):
            raise ValueError(f"Checkpoint {checkpoint[key]} is not the same as model {model_dict[key]}")
        
    return True