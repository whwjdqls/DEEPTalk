import torch
import json
import librosa
import numpy as np
from torch.utils.data._utils.collate import default_collate, default_collate_err_msg_format, np_str_obj_array_pattern, string_classes
import sys, os

from models import DEMOTE_VQ
import argparse
from utils.extra import seed_everything, compare_checkpoint_model
from utils.PyRenderMeshSequenceRenderer import PyRenderMeshSequenceRenderer, get_vertices_from_FLAME,save_video
from models.flame_models import flame
import glob
import tqdm
import imageio.v2 as imageio 


sys.path.append('../')
print(sys.path)
from DEE.get_DEE import get_DEE_from_json
from FER.get_model import init_affectnet_feature_extractor
from DEE.utils.utils import compare_checkpoint_model
# EMOTION_DICT = {'neutral': 1, 'calm': 2, 'happy': 3, 'sad': 4, 'angry' :  5, 'fear': 6, 'disgusted': 7, 'surprised': 8, 'contempt' : 9}
modify_DICT = {1:1, 3:2, 4:3, 5:7, 6:5, 7:6, 8:4, 9:8}
EMOTION_DICT = {'neutral': 1, 'happy': 2, 'sad': 3, 'surprised': 4, 'fear': 5, 'disgusted': 6, 'angry': 7, 'contempt': 8, 'calm': 9}
training_ids = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 
                'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 
                'M030', 'M031', 'W009', 'W011', 'W014', 'W015', 'W016', 'W018', 
                'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W028', 'W029'
                ] # 32 ids
MEAD_ACTOR_DICT = {k:i for i,k in enumerate(training_ids)}
    

def pad_audio_to_match_quantfactor(audio_samples, fps=30, quant_factor=3) :
    """padidng audio samples to be divisible by quant factor
    (NOTE) quant factor means the latents must be divisible by 2^(quant_factor)
           for inferno's EMOTE checkpoint, the quant_factor is 3 and fps is 25
    Args:
        audio_samples (torch tensor or numpy array): audio samples from raw wav files 
        fps (int, optional): fps of the face parameters. Defaults to 30.
        quant_factor (int, optional): squaushing latent variables by 2^(quant_factor) Defaults to 8.
    """
    if isinstance(audio_samples, np.ndarray):
        audio_samples = torch.tensor(audio_samples, dtype=torch.float32)
    
    audio_len = audio_samples.shape[0]
    latent_len = int(audio_len / 16000 * fps) # to target fps
    target_len = latent_len + (2**quant_factor - (latent_len % (2**quant_factor) )) # make sure the length is divisible by quant factor
    target_len = int(target_len / fps * 16000) # to audio sample rate

    padded_audio_samples = torch.nn.functional.pad(audio_samples, (0, target_len - len(audio_samples)))
    if isinstance(audio_samples, np.ndarray):
        padded_audio_samples = padded_audio_samples.numpy() 
    return padded_audio_samples


def main(args, config) :
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device', device)
    seed_everything(42)
    
    # loading FLINT checkpoint 
    FLINT_config_path = config['motionprior_config']['config_path']
    with open(FLINT_config_path, 'r') as f :
        FLINT_config = json.load(f) 
    # this actually does not matter because we are using load_motion_prior= Fale
    FLINT_ckpt = config['motionprior_config']['checkpoint_path']

    # Load model
    print("Loading Models...")
    DEE_config_path = glob.glob(f'{os.path.dirname(args.DEE_ckpt_path)}/*.json')[0]
    print(f'DEE config loaded :{DEE_config_path}') # DEE_config.use_affect_net
    DEE_model,DEE_config = get_DEE_from_json(DEE_config_path)
    DEE_checkpoint = torch.load(args.DEE_ckpt_path, map_location='cpu')
    DEE_model.load_state_dict(DEE_checkpoint)
    DEE_model.eval()
    compare_checkpoint_model(DEE_checkpoint, DEE_model.to('cpu'))

    affectnet_feature_extractor = None
    if DEE_config.affectnet_model_path:
        model_path = DEE_config.affectnet_model_path
        config_path = os.path.dirname(model_path) + '/config.yaml'
        _, affectnet_feature_extractor = init_affectnet_feature_extractor(config_path, model_path)
        affectnet_feature_extractor.to(device)
        affectnet_feature_extractor.eval()
        affectnet_feature_extractor.requires_grad_(False)
    # model = DEMOTE.DEMOTE(config, FLINT_config, FLINT_ckpt, DEE_model, load_motion_prior=config['motionprior_config']['load_motion_prior']).to(device)
    # model = DEMOTE_VQ.DEMOTE(config, FLINT_config,DEE_config, FLINT_ckpt, DEE_model, load_motion_prior=False)
    model = DEMOTE_VQ.DEMOTE_VQVAE_condition(config, FLINT_config,DEE_config, FLINT_ckpt, DEE_model, load_motion_prior=False)
    DEMOTE_ckpt_path = args.DEMOTE_ckpt_path
    DEMOTE_ckpt = torch.load(DEMOTE_ckpt_path, map_location='cpu')
    model.load_state_dict(DEMOTE_ckpt)
    ## check params
    state_dict = model.state_dict()
    DEE_checkpoint_ = torch.load(args.DEE_ckpt_path, map_location='cpu')
    FLINT_checkpoint_ = torch.load(FLINT_ckpt, map_location='cpu')
    print('*'*50)
    print('DEE')
    for key in DEE_checkpoint_.keys():
        if key.startswith('audio_encoder') :
            original_weights = DEE_checkpoint_[key]
            loaded_weights = state_dict[f'sequence_decoder.DEE.{key}']
            if not torch.allclose(original_weights, loaded_weights) :
                raise ValueError(f'{name} is different')
    print('*'*50)
    print('FLINT')
    for key in FLINT_checkpoint_.keys():
        if key.startswith('motion_decoder') :
            original_weights = FLINT_checkpoint_[key]
            new_key = key.replace('motion_decoder', 'sequence_decoder.motion_prior')
            loaded_weights = state_dict[new_key]
            if not torch.allclose(original_weights, loaded_weights) :
                raise ValueError(f'{name} is different')
    print('*'*50)
    print('Total')
    
    
    audio_path = args.audio_path
    for name, param in state_dict.items() :
        original_weights = DEMOTE_ckpt[name]
        if not torch.allclose(param, original_weights):
            raise ValueError(f'{name} is different')
    if audio_path.endswith('.wav') :
        wavdata, sampling_rate = librosa.load(audio_path, sr=16000)
    elif audio_path.endswith('.npy') :
        wavdata = np.load(audio_path)
    else:
        raise ValueError('audio file must be either .wav or .npy')
    
    

    save_dir = './outputs'
    os.makedirs(save_dir, exist_ok=True)
    param_save_dir = os.path.join(save_dir, 'params')
    os.makedirs(param_save_dir, exist_ok=True)
    file_name = os.path.basename(audio_path).split('.')[0]
    output_name = f'{file_name}'
    out_param_path = os.path.join(param_save_dir, f'{output_name}.npy')
    
    if not os.path.exists(out_param_path):
        audio = torch.tensor(wavdata, dtype=torch.float32)
        audio = pad_audio_to_match_quantfactor(audio, fps=config['audio_config']['target_fps'],
                                                quant_factor=config['sequence_decoder_config']['quant_factor'])

        emotion = 0 # this is random number as DEMOTE does not use this
        intensity = 0 # also a random number
        id = 'M003'
        actor_id = MEAD_ACTOR_DICT[id]
        n_emotions = config['sequence_decoder_config']["style_embedding"]['n_expression']
        n_intensities = config['sequence_decoder_config']["style_embedding"]['n_intensities']
        n_identities = config['sequence_decoder_config']["style_embedding"]['n_identities']
        condition_size = n_emotions + n_intensities + n_identities
        input_style = torch.eye(condition_size)[[emotion, # emotion one hot
                                                n_emotions + intensity, # intensity one hot
                                                n_emotions + n_intensities + actor_id]]  # actor id one hot
        
        input_style = torch.sum(input_style, dim=0).unsqueeze(0) # (1,60)
        inputs = []

        print(f'audio shape : {audio.shape}')
        audio = audio.unsqueeze(0) # (1, audio_len)
        model.eval()
        model.to(device)    
        audio = audio.to(device)
        input_style = input_style.to(device)
        print(f'audio : {audio.shape}')

        with torch.no_grad() :
                output = model(audio, input_style,sample=args.use_sampling,control_logvar=args.control_logvar,tau=args.tau)
                
        print(f'output : {output.shape}')
        
        # Batch, Frame_num, Params
        np.save(out_param_path, output.squeeze(0).detach().cpu().numpy())
        print(f'saved in {out_param_path}')
    else:
        output = torch.tensor(np.load(out_param_path)).unsqueeze(0).to(device)
        
    B,F,P = output.shape
    # save video
    video_save_dir = os.path.join(save_dir, 'videos')
    os.makedirs(video_save_dir, exist_ok=True)
    out_video_path = os.path.join(video_save_dir, f'{output_name}.mp4')
    
    print('initializing renderer...')
    template_mesh_path = './models/flame_models/FLAME_sample.ply'
    width = 800
    height = 800
    renderer = PyRenderMeshSequenceRenderer(template_mesh_path,
                                            width=width,
                                            height=height)
    print('getting flame model and calculating vertices...')
    expression_params = output[:,:, :50]
    jaw_pose = output[:,:, 50:53]

    FLAME = flame.FLAME(config, batch_size=1).to(device).eval()
    predicted_vertices = flame.get_vertices_from_flame(config, FLAME, expression_params, jaw_pose, device)
    predicted_vertices = predicted_vertices.reshape(B*F,-1,3) # (F,5023,3)
    print(f'vertices shape : {predicted_vertices.shape}')
    print('rendering...')
    T = len(predicted_vertices)
    pred_images = []
    for t in tqdm.tqdm(range(T)):
        pred_vertices = predicted_vertices[t].detach().cpu().view(-1,3).numpy()
        pred_image = renderer.render(pred_vertices)
        pred_images.append(pred_image) 
    
    print('saving video...')
    pred_images = np.stack(pred_images, axis=0)  
        
    save_video(out_video_path, pred_images, fourcc="mp4v", fps=25)
    if audio_path.endswith('.wav'):
        print('adding audio...')
        # audio_path = os.path.join(config.audio_path, f'{file_name}.wav')
        command = f'ffmpeg -y -i {out_video_path} -i {audio_path} -c:v copy -c:a aac -strict experimental {out_video_path.split(".mp4")[0]}_audio.mp4'
        os.system(command)
        os.remove(out_video_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DEMOTE_ckpt_path', type=str,required=True, help='path to EMOTE checkpoint, outputs will be saved in the same directory')
    parser.add_argument('--DEE_ckpt_path', type=str,required=True, help='path to DEE checkpoint')
    parser.add_argument('--audio_path', type=str,required=True, help='path to target audio')
    parser.add_argument('--use_sampling', action='store_true', help='sample from the emotion space to generate')
    parser.add_argument('--control_logvar', type=float, default=None, help='manually control logvariance of emotional sample space')
    parser.add_argument('--tau', type=float, default=0.0001, help='temperature for sampling')
    args = parser.parse_args()
    
    DEMOTE_config_path = f'{os.path.dirname(args.DEMOTE_ckpt_path)}/config.json'
    with open(DEMOTE_config_path) as f:
        DEMOTE_config = json.load(f)
        
    
        

    main(args, DEMOTE_config)

    