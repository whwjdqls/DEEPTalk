# from inferno.utils.other import get_path_to_externals
from pathlib import Path
import sys, os
import torch
import json
user_name = os.getcwd().split('/')[2]
sys.path.append(f'../')
from DEE.utils.pcme import sample_gaussian_tensors
# video emotion loss
# from inferno.models.temporal.Renderers import cut_mouth_vectorized
'''
E2E should be implemented from same version that LipReading used
'''
from externals.spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
## new lip reading loss
import torchvision.transforms as t
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import face_alignment

import argparse
import math
import numpy as np
import torch.nn as nn
import pickle
import omegaconf
## for video emotion loss
from omegaconf import OmegaConf
sys.path.append('models')
from video_emotion import SequenceClassificationEncoder, ClassificationHead, TransformerSequenceClassifier, MultiheadLinearClassificationHead, EmoCnnModule
import pytorch_lightning as pl 
from typing import Any, Optional, Dict, List
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision
from scipy.signal import find_peaks

def check_nan(sample: Dict): 
    ok = True
    nans = []
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN found in '{key}'")
                nans.append(key)
                ok = False
                # raise ValueError("Nan found in sample")
    if len(nans) > 0:
        raise ValueError(f"NaN found in {nans}")
    return ok

# remove training part of VideoClassification
class VideoClassifierBase(pl.LightningModule): 

    def __init__(self, 
                 cfg, 
                 preprocessor = None,
                 feature_model = None,
                 fusion_layer: Optional[nn.Module] = None,
                 sequence_encoder: Optional[SequenceClassificationEncoder] = None,
                 classification_head: Optional[ClassificationHead] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.feature_model = feature_model
        self.fusion_layer = fusion_layer
        self.sequence_encoder = sequence_encoder
        self.classification_head = classification_head

    def get_trainable_parameters(self):
        trainable_params = []
        if self.feature_model is not None:
            trainable_params += self.feature_model.get_trainable_parameters()
        if self.sequence_encoder is not None:
            trainable_params += self.sequence_encoder.get_trainable_parameters()
        if self.classification_head is not None:
            trainable_params += self.classification_head.get_trainable_parameters()
        return trainable_params

    @property
    def max_seq_length(self):
        return 5000

    @torch.no_grad()
    def preprocess_input(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        if self.preprocessor is not None:
            if self.device != self.preprocessor.device:
                self.preprocessor.to(self.device)
            sample = self.preprocessor(sample, input_key="video", train=train, test_time=not train, **kwargs)
        # sample = detach_dict(sample)
        return sample 

    def is_multi_modal(self):
        modality_list = self.cfg.model.get('modality_list', None) 
        return modality_list is not None and len(modality_list) > 1

    def forward(self, sample: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        """
        sample: Dict[str, torch.Tensor]
            - gt_emo_feature: (B, T, F)
        """
        # T = sample[input_key].shape[1]
        if "gt_emo_feature" in sample:
            T = sample['gt_emo_feature'].shape[1]
        else: 
            T = sample['video'].shape[1]
        if self.max_seq_length < T: # truncate
            print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
            sample = truncate_sequence_batch(sample, self.max_seq_length)

        # preprocess input (for instance get 3D pseudo-GT )
        sample = self.preprocess_input(sample, train=train, **kwargs)
        check_nan(sample)

        if self.feature_model is not None:
            sample = self.feature_model(sample, train=train, **kwargs)
            check_nan(sample)
        else:
            input_key = "gt_emo_feature" # TODO: this needs to be redesigned 
            sample["hidden_feature"] = sample[input_key]

        if self.sequence_encoder is not None:
            sample = self.sequence_encoder(sample) #, train=train, validation=validation, **kwargs)
            check_nan(sample)

        if self.classification_head is not None:
            sample = self.classification_head(sample)
            check_nan(sample)

        return sample

    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoClassifierBase':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = VideoClassifierBase(cfg, prefix)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = VideoClassifierBase.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                strict=False, 
                **checkpoint_kwargs)
        return model
        
def truncate_sequence_batch(sample: Dict, max_seq_length: int) -> Dict:
    """
    Truncate the sequence to the given length. 
    """
    # T = sample["audio"].shape[1]
    # if max_seq_length < T: # truncate
    for key in sample.keys():
        if isinstance(sample[key], torch.Tensor): # if temporal element, truncate
            if sample[key].ndim >= 3:
                sample[key] = sample[key][:, :max_seq_length, ...]
        elif isinstance(sample[key], Dict): 
            sample[key] = truncate_sequence_batch(sample[key], max_seq_length)
        elif isinstance(sample[key], List):
            pass
        else: 
            raise ValueError(f"Invalid type '{type(sample[key])}' for key '{key}'")
    return sample

class VideoEmotionClassifier(VideoClassifierBase): 

    def __init__(self, 
                 cfg
        ):
        self.cfg = cfg
        preprocessor = None
        # feature_model = feature_enc_from_cfg(cfg.model.get('feature_extractor', None))
        feature_model = None
        fusion_layer = None
        if not self.is_multi_modal():
            feature_size = feature_model.output_feature_dim() if feature_model is not None else cfg.model.input_feature_size
        sequence_classifier = TransformerSequenceClassifier(cfg.model.get('sequence_encoder', None), feature_size)
        classification_head = MultiheadLinearClassificationHead(cfg.model.get('classification_head', None), 
                                                           sequence_classifier.encoder_output_dim(), 
                                                           cfg.model.output.num_classes,
                                                           )

        super().__init__(cfg,
            preprocessor = preprocessor,
            feature_model = feature_model,
            fusion_layer = fusion_layer,
            sequence_encoder = sequence_classifier,  
            classification_head = classification_head,  
        )


    @classmethod
    def instantiate(cls, cfg, stage, prefix, checkpoint, checkpoint_kwargs) -> 'VideoEmotionClassifier':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = VideoEmotionClassifier(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = VideoEmotionClassifier.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **checkpoint_kwargs
            )

        return model

def locate_checkpoint(cfg_or_checkpoint_dir, replace_root = None, relative_to = None, mode=None, pattern=None):
    if isinstance(cfg_or_checkpoint_dir, (str, Path)):
        checkpoint_dir = str(cfg_or_checkpoint_dir)
    elif replace_root is not None:
        checkpoint_dir = str(Path(replace_root) / Path(cfg_or_checkpoint_dir.inout.checkpoint_dir))
    else :
        checkpoint_dir = cfg_or_checkpoint_dir.inout.checkpoint_dir

    if replace_root is not None and relative_to is not None:
        try:
            checkpoint_dir = str(Path(replace_root) / Path(checkpoint_dir).relative_to(relative_to))
        except ValueError as e:
            print(f"Not replacing the root of checkpoint_dir '{checkpoint_dir}' beacuse the specified root does not fit:"
                  f"'{replace_root}'")

    print(f"Looking for checkpoint in '{checkpoint_dir}'")
    checkpoints = sorted(list(Path(checkpoint_dir).rglob("*.ckpt")))
    if len(checkpoints) == 0:
        print(f"Did not find checkpoints. Looking in subfolders")
        checkpoints = sorted(list(Path(checkpoint_dir).rglob("*.ckpt")))
        if len(checkpoints) == 0:
            print(f"Did not find checkpoints to resume from. Returning None")
            # sys.exit()
            return None
        print(f"Found {len(checkpoints)} checkpoints")
    else:
        print(f"Found {len(checkpoints)} checkpoints")
    if pattern is not None:
        checkpoints = [ckpt for ckpt in checkpoints if pattern in str(ckpt)]
    for ckpt in checkpoints:
        print(f" - {str(ckpt)}")

    if isinstance(mode, int):
        checkpoint = str(checkpoints[mode])
    elif mode == 'latest':
        # checkpoint = str(checkpoints[-1])
        checkpoint = checkpoints[0]
        # assert checkpoint.name == "last.ckpt", f"Checkpoint name is not 'last.ckpt' but '{checkpoint.name}'. Are you sure this is the right checkpoint?"
        if checkpoint.name != "last.ckpt":
            # print(f"Checkpoint name is not 'last.ckpt' but '{checkpoint.name}'. Are you sure this is the right checkpoint?")
            return None
        checkpoint = str(checkpoint)
    elif mode == 'best':
        min_value = 999999999999999.
        min_idx = -1
        # remove all checkpoints that do not containt the pattern 
        for idx, ckpt in enumerate(checkpoints):
            if ckpt.stem == "last": # disregard last
                continue
            end_idx = str(ckpt.stem).rfind('=') + 1
            loss_str = str(ckpt.stem)[end_idx:]
            try:
                loss_value = float(loss_str)
            except ValueError as e:
                print(f"Unable to convert '{loss_str}' to float. Skipping this checkpoint.")
                continue
            if loss_value <= min_value:
                min_value = loss_value
                min_idx = idx
        if min_idx == -1:
            raise FileNotFoundError("Finding the best checkpoint failed")
        checkpoint = str(checkpoints[min_idx])
    else:
        raise ValueError(f"Invalid checkpoint loading mode '{mode}'")
    print(f"Selecting checkpoint '{checkpoint}'")
    return checkpoint

def get_checkpoint_with_kwargs(cfg, prefix, replace_root = None, relative_to = None, checkpoint_mode=None, pattern=None):
    checkpoint = locate_checkpoint(cfg, replace_root = replace_root,
                                   relative_to = relative_to, mode=checkpoint_mode, pattern=pattern)
    cfg.model.resume_training = False  # make sure the training is not magically resumed by the old code
    checkpoint_kwargs = {'config': cfg}
    return checkpoint, checkpoint_kwargs

def class_from_str(str, module=None, none_on_fail = False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")

def emo_network_from_path(path):
    print(f"Loading trained emotion network from: '{path}'")

    def load_configs(run_path):
        from omegaconf import OmegaConf
        with open(Path(run_path) / "cfg.yaml", "r") as f:
            conf = OmegaConf.load(f)
        if run_path != conf.inout.full_run_dir: 
            conf.inout.output_dir = str(Path(run_path).parent)
            conf.inout.full_run_dir = str(run_path)
            conf.inout.checkpoint_dir = str(Path(run_path) / "checkpoints")
        return conf

    cfg = load_configs(path)

    if not bool(cfg.inout.checkpoint_dir):
        cfg.inout.checkpoint_dir = str(Path(path) / "checkpoints")

    checkpoint_mode = 'best'
    stages_prefixes = ""

    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, stages_prefixes,
                                                               checkpoint_mode=checkpoint_mode,
                                                               )
    checkpoint_kwargs = checkpoint_kwargs or {}

    if 'emodeca_type' in cfg.model.keys():
        module_class = class_from_str(cfg.model.emodeca_type, sys.modules[__name__])
    else:
        module_class = EmoNetModule

    #module_class = EmoCnnModule
    emonet_module = module_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False,
                                                      **checkpoint_kwargs)
    return emonet_module

def metric_from_str(metric, **kwargs) :
    if metric == 'cosine_similarity' :
        return cosine_sim_negative

def cosine_sim_negative(*args, **kwargs) :
    return (1. - F.cosine_similarity(*args, **kwargs)).mean()


def create_video_emotion_loss(cfg): # cfg = EMOTE_config['loss']['emotion_video_loss']
    model_config_path = Path(cfg["network_path"]) / "cfg.yaml"
    # load config 
    model_config = OmegaConf.load(model_config_path)

    # sequence_model = load_video_emotion_recognition_net(cfg.network_path)
    class_ = class_from_str(model_config.model.pl_module_class, sys.modules[__name__])

    # instantiate the model
    checkpoint_mode = 'best' # resuming in the same stage, we want to pick up where we left of
    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(
        model_config, "", replace_root = 'models/VideoEmotionRecognition/models',
        checkpoint_mode=checkpoint_mode,
        pattern="val"
        )

    sequence_model = class_.instantiate(model_config, None, None, checkpoint, checkpoint_kwargs)

    ## see if the model has a feature extractor
    feat_extractor_cfg = model_config.model.get('feature_extractor', None)
     
    if (feat_extractor_cfg is None or feat_extractor_cfg["type"] is False) and cfg['feature_extractor_path']:
        # default to the affecnet trained resnet feature extractor
        feature_extractor_path = Path(cfg["feature_extractor_path"])
        feature_extractor = emo_network_from_path(str(feature_extractor_path))
    elif cfg["feature_extractor"] == "no":
        feature_extractor = None
    else: 
        feature_extractor = None
    
    metric = metric_from_str(cfg['metric'])
    loss = VideoEmotionRecognitionLoss(sequence_model, metric, feature_extractor)
    return loss

class VideoEmotionRecognitionLoss(torch.nn.Module):

    def __init__(self, video_emotion_recognition : VideoEmotionClassifier, metric, feature_extractor=None, ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.video_emotion_recognition = video_emotion_recognition
        self.metric = metric

    def forward(self, input, target):
        raise NotImplementedError()

    def _forward_input(self, 
        input_images=None, 
        input_emotion_features=None,
        mask=None,
        return_logits=False,
        ):
        with torch.no_grad():
            return self.forward(input_images, input_emotion_features, mask, return_logits)

    def _forward_output(self, 
        output_images=None, 
        output_emotion_features=None,
        mask=None,
        return_logits=False,
        ):
        return self.forward(output_images, output_emotion_features, mask, return_logits)

    def forward(self, 
        images=None, 
        emotion_features=None,
        mask=None,
        return_logits=False,
        ):
        assert images is not None or emotion_features is not None, \
            "One and only one of input_images or input_emotion_features must be provided"
        if images is not None:
            B, T = images.shape[:2]
        else: 
            B, T = emotion_features.shape[:2]
        if emotion_features is None:
            feat_extractor_sample = {"image" : images.view(B*T, *images.shape[2:])}
            emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)
            # result_ = self.model.forward_old(images)
        if mask is not None:
            emotion_features = emotion_features * mask

        video_emorec_batch = {
            "gt_emo_feature": emotion_features,
        }
        video_emorec_batch = self.video_emotion_recognition(video_emorec_batch)

        emotion_feat = video_emorec_batch["pooled_sequence_feature"]
        
        if return_logits:
            if "predicted_logits" in video_emorec_batch:
                predicted_logits = video_emorec_batch["predicted_logits"]
                return emotion_feat, predicted_logits
            logit_list = {}
            if "predicted_logits_expression" in video_emorec_batch:
                logit_list["predicted_logits_expression"] = video_emorec_batch["predicted_logits_expression"]
            if "predicted_logits_intensity" in video_emorec_batch:
                logit_list["predicted_logits_intensity"] = video_emorec_batch["predicted_logits_intensity"]
            if "predicted_logits_identity" in video_emorec_batch:
                logit_list["predicted_logits_identity"] = video_emorec_batch["predicted_logits_identity"]
            return emotion_feat, logit_list

        return emotion_feat


    def compute_loss(
        self, 
        input_images=None, 
        input_emotion_features=None,
        output_images=None, 
        output_emotion_features=None,
        mask=None, 
        return_logits=False,
        ):
        # assert input_images is not None or input_emotion_features is not None, \
        #     "One and only one of input_images or input_emotion_features must be provided"
        # assert output_images is not None or output_emotion_features is not None, \
        #     "One and only one of output_images or output_emotion_features must be provided"
        # # assert mask is None, "Masked loss not implemented for video emotion recognition"

        if return_logits:
            input_emotion_feat, in_logits = self._forward_input(input_images, input_emotion_features, mask, return_logits=return_logits)
            output_emotion_feat, out_logits = self._forward_output(output_images, output_emotion_features, mask, return_logits=return_logits)
            return self._compute_feature_loss(input_emotion_feat, output_emotion_feat), in_logits, out_logits

        input_emotion_feat = self._forward_input(input_images, input_emotion_features, mask)
        output_emotion_feat = self._forward_output(output_images, output_emotion_features, mask)
        return self._compute_feature_loss(input_emotion_feat, output_emotion_feat)


    def _compute_feature_loss(self, input_emotion_feat, output_emotion_feat):
        loss = self.metric(input_emotion_feat, output_emotion_feat)

        return loss
    


class LipReadingNet(torch.nn.Module):

    def __init__(self, loss_cfg, device): 
        super().__init__()

        model_path = loss_cfg['lip_reading_loss']['E2E']['model']['model_path']
        model_conf = loss_cfg['lip_reading_loss']['E2E']['model']['model_conf']

        with open(model_conf, 'rb') as f:
            confs = json.load(f)
        if isinstance(confs, dict):
            args = confs
        else :
            idim, odim, args = confs
            self.odim = odim
        self.train_args = argparse.Namespace(**args)

        self.lip_reader = E2E(odim, self.train_args)
        self.lip_reader.load_state_dict(torch.load(model_path))

        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        self.mouth_transform = t.Compose([
            t.Normalize(0.0, 1.0),
            t.CenterCrop(crop_size),
            t.Normalize(mean, std)]
        )


    def forward(self, lip_images):
        """
        :param lip_images: (batch_size, seq_len, 1, 88, 88) or (seq_len, 1, 88, 88))
        """
        # this is my - hopefully fixed version of the forward pass
        # In other words, in the lip reading repo code, the following happens:
        # inferno/external/spectre/external/Visual_Speech_Recognition_for_Multiple_Languages/espnet/nets/pytorch_backend/backbones/conv3d_extractor.py
        # line 95:
        # B, C, T, H, W = xs_pad.size() # evaluated to: torch.Size([B, 1, 70, 88, 88]) - so the temporal window is collapsed into the batch size

        ndim = lip_images.ndim
        B, T = lip_images.shape[:2]
        rest = lip_images.shape[2:]
        if ndim == 5: # batched 
            lip_images = lip_images.view(B * T, *rest)
        elif ndim == 4: # single
            pass
        else: 
            raise ValueError("Lip images should be of shape (batch_size, seq_len, 1, 88, 88) or (seq_len, 1, 88, 88)")

        channel_dim = 1
        lip_images = self.mouth_transform(lip_images.squeeze(channel_dim)).unsqueeze(channel_dim)

        if ndim == 5:
            lip_images = lip_images.view(B, T, *lip_images.shape[2:])
        elif ndim == 4: 
            lip_images = lip_images.unsqueeze(0)
            lip_images = lip_images.squeeze(2)

        # the image is now of shape (B, T, 88, 88), the missing channel dimension is unsqueezed in the lipread net code
        # lip_features = self.lip_reader.model.encoder(
        # (JB/12-10)ECE net seems to not have a model class, but the encoder is a module in the E2E class

        lip_features = self.lip_reader.encoder(
            lip_images,
            None,
            extract_resnet_feats=True
        )
        return lip_features

class LipReadingLoss(torch.nn.Module):
    '''
    Use LipReading Loss via LipReadingLoss(mout_gt, mout_pred)
    '''

    def __init__(self, loss_cfg, device, loss='cosine_similarity', 
                mouth_crop_width = 96,
                mouth_crop_height = 96,
                mouth_window_margin = 12,
                mouth_landmark_start_idx = 48,
                mouth_landmark_stop_idx = 68,
                ):
        super().__init__()
        self.loss = loss
        assert loss in ['cosine_similarity', 'l1_loss', 'mse_loss']
        self.model = LipReadingNet(loss_cfg, device)
        self.model.eval()
        # freeze model
        for param in self.parameters(): 
            param.requires_grad = False

        self.mouth_crop_width = mouth_crop_width
        self.mouth_crop_height = mouth_crop_height
        self.mouth_window_margin = mouth_window_margin
        self.mouth_landmark_start_idx = mouth_landmark_start_idx
        self.mouth_landmark_stop_idx = mouth_landmark_stop_idx

    def _forward_input(self, images):
        # there is no need to keep gradients for input (even if we're finetuning, which we don't, it's the output image we'd wannabe finetuning on)
        with torch.no_grad():
            result = self.model(images)
        return result

    def _forward_output(self, images):
        return self.model(images)
    
    def forward(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)

    def compute_loss(self, mouth_images_gt, mouth_images_pred, mask=None):
        lip_features_gt = self._forward_input(mouth_images_gt)
        lip_features_pred = self._forward_output(mouth_images_pred)

        lip_features_gt = lip_features_gt.view(-1, lip_features_gt.shape[-1])
        lip_features_pred = lip_features_pred.view(-1, lip_features_pred.shape[-1])
        
        if mask is not None:
            lip_features_gt = lip_features_gt[mask.view(-1)]
            lip_features_pred = lip_features_pred[mask.view(-1)]
        
        return self._compute_feature_loss(lip_features_gt, lip_features_pred)

    def _compute_feature_loss(self, lip_features_gt, lip_features_pred): 
        if self.loss == 'cosine_similarity':
            # pytorch cosine similarity
            lr = 1-torch.nn.functional.cosine_similarity(lip_features_gt, lip_features_pred, dim=1).mean()
            ## manual cosine similarity  take over from spectre
            # lr = (lip_features_gt*lip_features_pred).sum(1)/torch.linalg.norm(lip_features_pred,dim=1)/torch.linalg.norm(lip_features_gt,dim=1)
            # lr = 1 - lr.mean()
        elif self.loss == 'l1_loss':
            lr = torch.nn.functional.l1_loss(lip_features_gt, lip_features_pred)
        elif self.loss == 'mse_loss':
            lr = torch.nn.functional.mse_loss(lip_features_gt, lip_features_pred)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
        return lr
    
    def _compute_jaw_peaks_weight_loss(self, lip_features_gt, lip_features_pred, jaw_gt) :
        BS,T,_ = jaw_gt.shape
        _,emb = lip_features_gt.shape
        if type(jaw_gt) == torch.Tensor:
            jaw_gt = jaw_gt.cpu().numpy()
        mask = torch.zeros(jaw_gt.shape[:2])
        
        for i,jaw_gt_instance in enumerate(jaw_gt[:,:,0]):
            peaks, _ = find_peaks(jaw_gt_instance, distance=5,prominence=0.02)
            low_peaks, _ = find_peaks(-jaw_gt_instance, distance=5, prominence=0.02)
            mask[i][peaks] = 1
            mask[i][low_peaks] = 1
            
        target_pred_cosine_sim = F.cosine_similarity(lip_features_gt, lip_features_pred, dim=1)
        mask = mask.float().view(BS*T).to(lip_features_gt.device)
        lip_loss = 1 - torch.sum(target_pred_cosine_sim * mask) / torch.sum(mask)
        return lip_loss

    
class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)

class new_LipReadingLoss(torch.nn.Module) :
    def __init__(self, device, lip_reading_model, mouth_window_margin, mouth_landmark_start_idx,
                mouth_landmark_stop_idx, mouth_crop_height, mouth_crop_width,
                convert_grayscale=True) :
        super().__init__()
        self.mouth_window_margin = mouth_window_margin
        self.mouth_landmark_start_idx = mouth_landmark_start_idx
        self.mouth_landmark_stop_idx = mouth_landmark_stop_idx
        self.mouth_crop_height = mouth_crop_height
        self.mouth_crop_width = mouth_crop_width
        self.convert_grayscale = convert_grayscale
        self.transform = t.Compose([t.Resize((224,224))])
        self.mouth_transform_ = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        self.device = device

        self.landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='sfd', device=str(self.device))
        self.lip_reading_model = lip_reading_model


    def forward(self, images) :
        '''
        landmark shape : (BS,T,68,2)
        image shape : (BS,T,3,H,W)
        '''
        if images.shape[3] > 224 :
            frames = self.transform(images)
        else :
            frames = images
        frames = frames*255.0
        landmarks = self.detect_landmarks(frames)

        mouth_video = self.cut_mouth_vectorized(frames, landmarks)
        mouth_video = self.mouth_transform_(mouth_video)

        BS,T,C,H,W = mouth_video.shape
        mouth_video = mouth_video.reshape(BS*T,C,H,W)
        embedding = self.lip_reading_model.encoder(mouth_video, masks=None, extract_resnet_feats=True)
        embedding = embedding.reshape(BS,T,-1) # (BS,T,512)

        return embedding
    
    def compute_feature_loss(self, lip_features_gt, lip_features_pred): 
        BS,T,emb = lip_features_gt.shape
        lip_features_gt = lip_features_gt.view(BS*T, -1)
        lip_features_pred = lip_features_pred.view(BS*T, -1)
        loss = 1-F.cosine_similarity(lip_features_gt, lip_features_pred, dim=1).mean()

        return loss
    
    def compute_anchor_feature_loss(self, lip_features_pred, anchor, threshold, target_lip_features, compute_with_anchor) :
        BS,T,emb = target_lip_features.shape
        anchor = anchor.repeat(BS*T,1)
        target_lip_features = target_lip_features.view(BS*T,-1)
        lip_features_pred = lip_features_pred.view(BS*T,-1)

        cosine_sim = F.cosine_similarity(target_lip_features, anchor, dim=1) #BS*T
        weight_mask = cosine_sim > threshold
        weights = weight_mask.float()
        if compute_with_anchor :
            anchor_pred_cosine_sim = F.cosine_similarity(lip_features_pred, anchor, dim=1)
            weighted_sum = torch.sum(anchor_pred_cosine_sim * weights)
        else :
            target_pred_cosine_sim = F.cosine_similarity(lip_features_pred, target_lip_features, dim=1) #BS*T
            # masking for only closed lips
            weighted_sum = torch.sum(target_pred_cosine_sim * weights)

        if torch.sum(weights) == 0 :
            print('No closed lips')
            return weighted_sum

        # compute mean of closed lips
        weighted_sum /= torch.sum(weights)
        loss = 1-weighted_sum

        return loss

    def compute_jaw_filter_loss(self, lip_features_pred, target_lip_features, jaw_gt, threshold) :
        BS,T,emb = target_lip_features.shape
        target_lip_features = target_lip_features.view(BS*T,-1)
        lip_features_pred = lip_features_pred.view(BS*T,-1)
        
        weight_mask = jaw_gt[:,:,0] > threshold # (BS,T,3)
        weights = weight_mask.float()
        weights = weights.view(BS*T,-1).squeeze(1)

        target_pred_cosine_sim = F.cosine_similarity(lip_features_pred, target_lip_features, dim=1)
        # masking for only closed jaws
        weighted_sum = torch.sum(target_pred_cosine_sim * weights)

        if torch.sum(weights) == 0 :
            print('No closed jaws')
            return weighted_sum
        
        # compute mean of closed jaws
        weighted_sum /= torch.sum(weights)
        loss = 1-weighted_sum

        return loss

    def detect_landmarks(self, images) :
        BS,T,C,H,W = images.shape
        images = images.reshape(BS*T,C,H,W)

        landmarks = self.landmarks_detector.get_landmarks_from_batch(images.to(self.device, dtype=torch.float32))
        landmarks = np.array(landmarks)
        landmarks = torch.tensor(landmarks).to(self.device)
        landmarks = landmarks.reshape(BS,T,68,2)

        return landmarks

    def cut_mouth_vectorized(self, images, landmarks) :
        with torch.no_grad():
            image_size = images.shape[-1] / 2

            landmarks = landmarks * image_size + image_size
            # #1) smooth the landmarks with temporal convolution
            # landmarks are of shape (T, 68, 2) 
            # reshape to (T, 136) 
            landmarks_t = landmarks.reshape(*landmarks.shape[:2], -1) # (BS,T,136)

            # make temporal dimension last 
            landmarks_t = landmarks_t.permute(0, 2, 1) # (BS,136,T)
            # smooth with temporal convolution
            temporal_filter = torch.ones(self.mouth_window_margin, device=images.device) / self.mouth_window_margin
            # pad the the landmarks 
            landmarks_t_padded = F.pad(landmarks_t, (self.mouth_window_margin // 2, self.mouth_window_margin // 2), mode='replicate')
            # convolve each channel separately with the temporal filter
            num_channels = landmarks_t.shape[1]
            smooth_landmarks_t = F.conv1d(landmarks_t_padded, 
                temporal_filter.unsqueeze(0).unsqueeze(0).expand(num_channels,1,temporal_filter.numel()), 
                groups=num_channels, padding='valid'
            )
            smooth_landmarks_t = smooth_landmarks_t[..., 0:landmarks_t.shape[-1]]

            # reshape back to the original shape 
            smooth_landmarks_t = smooth_landmarks_t.permute(0, 2, 1).view(landmarks.shape)
            smooth_landmarks_t = smooth_landmarks_t + landmarks.mean(dim=2, keepdims=True) - smooth_landmarks_t.mean(dim=2, keepdims=True)

            # #2) get the mouth landmarks
            mouth_landmarks_t = smooth_landmarks_t[..., self.mouth_landmark_start_idx:self.mouth_landmark_stop_idx, :]
            # print(f'mouth_landmark : {mouth_landmarks_t}')
            
            # #3) get the mean of the mouth landmarks
            mouth_landmarks_mean_t = mouth_landmarks_t.mean(dim=-2, keepdims=True)
        
            # #4) get the center of the mouth
            center_x_t = mouth_landmarks_mean_t[..., 0]
            center_y_t = mouth_landmarks_mean_t[..., 1]

            # #5) use grid_sample to crop the mouth in every image 
            # create the grid
            height = self.mouth_crop_height//2
            width = self.mouth_crop_width//2

            torch.arange(0, self.mouth_crop_width, device=images.device)

            grid = torch.stack(torch.meshgrid(torch.linspace(-height, height, self.mouth_crop_height).to(images.device) / (images.shape[-2] /2),
                                    torch.linspace(-width, width, self.mouth_crop_width).to(images.device) / (images.shape[-1] /2) ), 
                                    dim=-1)
            grid = grid[..., [1, 0]]
            grid = grid.unsqueeze(0).unsqueeze(0).repeat(*images.shape[:2], 1, 1, 1)

            center_x_t -= images.shape[-1] / 2
            center_y_t -= images.shape[-2] / 2

            center_x_t /= images.shape[-1] / 2 # 112
            center_y_t /= images.shape[-2] / 2 # 141

            center_xy =  torch.cat([center_x_t, center_y_t ], dim=-1).unsqueeze(-2).unsqueeze(-2)
            if center_xy.ndim != grid.ndim:
                center_xy = center_xy.unsqueeze(-2)
            assert grid.ndim == center_xy.ndim, f"grid and center_xy have different number of dimensions: {grid.ndim} and {center_xy.ndim}"

            # grid should be in range [-1,1]
            half_height = images.shape[-1] / 2
            grid = grid + ((center_xy-half_height)/half_height)

        B, T = images.shape[:2]
        images = images.view(B*T, *images.shape[2:]) # (BS*T,3,224,224)
        grid = grid.view(B*T, *grid.shape[2:]) # (BS*T,224,224,2)

        if self.convert_grayscale: 
            images = TF.rgb_to_grayscale(images)

        image_crops = F.grid_sample(
            images, 
            grid,  
            align_corners=True, 
            padding_mode='zeros',
            mode='bicubic'
            )
        image_crops = image_crops.view(B, T, *image_crops.shape[1:])

        if self.convert_grayscale:
            image_crops = image_crops#.squeeze(1)

        return image_crops



# Following EMOTE paper,
# λrec is set to 1000000 and λKL to 0.001, which makes the
# converged KL divergence term less than one order of magnitude
# lower than the reconstruction terms
def calc_vae_loss(pred,target,mu, logvar, recon_weight=1_000_000, kl_weight=0.001):                            
    """ function that computes the various components of the VAE loss """
    reconstruction_loss = nn.MSELoss()(pred, target)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return recon_weight * reconstruction_loss + kl_weight * KLD, recon_weight *reconstruction_loss,kl_weight * KLD


def calc_vq_loss(pred, target, quant_loss, quant_loss_wight,alpha=1.0):
    """ function that computes the various components of the VQ loss """

    exp_loss = nn.L1Loss()(pred[:,:,:50], target[:,:,:50])
    rot_loss = nn.L1Loss()(pred[:,:,50:53], target[:,:,50:53])
    jaw_loss = alpha * nn.L1Loss()(pred[:,:,53:], target[:,:,53:])
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean()*quant_loss_wight + \
            (exp_loss + rot_loss + jaw_loss)

def calc_vq_flame_L1_loss(pred, target, quant_loss,quant_loss_wight=1.0, alpha=1.0):
    exp_loss = nn.L1Loss()(pred[:,:,:50], target[:,:,:50])
    jaw_loss = alpha * nn.L1Loss()(pred[:,:,50:53], target[:,:,50:53])
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean()*quant_loss_wight + \
            (exp_loss + jaw_loss), (exp_loss + jaw_loss)
            
def calc_vq_flame_L2_loss(pred, target, quant_loss, quant_loss_wight=1.0,alpha=1.0):
    exp_loss = nn.MSELoss()(pred[:,:,:50], target[:,:,:50])
    jaw_loss = alpha * nn.MSELoss()(pred[:,:,50:53], target[:,:,50:53])
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean()*quant_loss_wights + \
            (exp_loss + jaw_loss), (exp_loss + jaw_loss)
            
def calc_vq_vertice_L2_loss(pred, target, quant_loss, quant_loss_wight=1.0, recon_weight=1000000):
    reconstruction_loss = nn.MSELoss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean()*quant_loss_wight + reconstruction_loss*recon_weight  , reconstruction_loss*recon_weight

def calculate_vertice_loss(pred, target):
     reconstruction_loss = nn.MSELoss()(pred, target)
     return reconstruction_loss

def calculate_vertex_velocity_loss(pred, target):
    """
    pred, target torch tensor of shape (BS, T, V*3)
    """
    pred_diff = pred[:, 1:,:] - pred[:, :-1,:]
    target_diff = target[:, 1:,:] - target[:, :-1,:]
    velocity_loss = nn.MSELoss()(pred_diff, target_diff)
    return velocity_loss


def calculate_consistency_loss(model, audio, pred_exp, 
                               point_DEE, normalize_exp=True,
                               num_samples = None, affectnet_feature_extractor=None,
                               prob_method='csd') :
    if normalize_exp :
        pred_exp = (pred_exp - torch.mean(pred_exp, dim=1, keepdim=True)) / torch.std(pred_exp, dim=1, keepdim=True)
    if point_DEE :

        DEE_audio_embedding = model.encode_audio(audio)
        if affectnet_feature_extractor is not None:
            pred_exp = affectnet_feature_extractor.extract_feature_from_layer(pred_exp, layer_num = -2)
        DEE_exp_embedding = model.encode_parameter(pred_exp)
        # Normalize before computing cosine similarity
        DEE_audio_embedding /= DEE_audio_embedding.norm(dim=1, keepdim=True)
        DEE_exp_embedding /= DEE_exp_embedding.norm(dim=1, keepdim=True)
        emotion_consistency_loss = 1-F.cosine_similarity(DEE_audio_embedding, DEE_exp_embedding, dim=1).mean()
        return emotion_consistency_loss
    else :

        audio_mean, audio_logvar = model.audio_encoder(audio)
        if affectnet_feature_extractor is not None:
            pred_exp = affectnet_feature_extractor.extract_feature_from_layer(pred_exp, layer_num = -2)
        exp_mean, exp_logvar = model.exp_encoder(pred_exp)
        # sampling
        if prob_method == 'csd' :
            mu_pdist = ((audio_mean - exp_mean) ** 2).sum(-1)
            sigma_pdist = ((torch.exp(audio_logvar) + torch.exp(exp_logvar))).sum(-1)
            logits = mu_pdist + sigma_pdist
            logits = -model.criterion.negative_scale * logits + model.criterion.shift # (Bs, BS)
            labels = torch.ones(logits.shape[0], dtype=logits.dtype, device=logits.device)
            loss = model.criterion.bceloss(logits, labels)
            return loss
        elif prob_method == 'mean':
            mu_pdist = ((audio_mean - exp_mean) ** 2).sum(-1)
            return mu_pdist.mean()
        
        elif prob_method == 'sample' :
            raise NotImplementedError
            DEE_audio_embedding = sample_gaussian_tensors(audio_mean, audio_logvar, num_samples, normalize=True)
            DEE_exp_embedding = sample_gaussian_tensors(exp_mean, exp_logvar, num_samples, normalize=True)
            if num_samples == 1 :
                DEE_audio_embedding = DEE_audio_embedding.squeeze(1)
                DEE_exp_embedding = DEE_exp_embedding.squeeze(1)
    raise ValueError(f"Invalid prob_method '{prob_method}'")

def calculate_VV_emo_loss(model, target_exp, pred_exp, 
                               point_DEE, normalize_exp=True,
                               num_samples = None, affectnet_feature_extractor=None,
                               prob_method='csd'):
    if normalize_exp :
        pred_exp = (pred_exp - torch.mean(pred_exp, dim=1, keepdim=True)) / torch.std(pred_exp, dim=1, keepdim=True)
        target_exp = (target_exp - torch.mean(target_exp, dim=1, keepdim=True)) / torch.std(target_exp, dim=1, keepdim=True)
    if point_DEE :
        if affectnet_feature_extractor is not None:
            pred_exp = affectnet_feature_extractor.extract_feature_from_layer(pred_exp, layer_num = -2)
            target_exp = affectnet_feature_extractor.extract_feature_from_layer(target_exp, layer_num = -2)
        pred_embedding = model.encode_parameter(pred_exp)
        target_embedding = model.encode_parameter(target_exp)
        # Normalize before computing cosine similarity
        pred_embedding /= pred_embedding.norm(dim=1, keepdim=True)
        target_embedding /= target_embedding.norm(dim=1, keepdim=True)
        emotion_consistency_loss = 1-F.cosine_similarity(pred_embedding, target_embedding, dim=1).mean()
        return emotion_consistency_loss
    else :
        if affectnet_feature_extractor is not None:
            pred_exp = affectnet_feature_extractor.extract_feature_from_layer(pred_exp, layer_num = -2)
            target_exp = affectnet_feature_extractor.extract_feature_from_layer(target_exp, layer_num = -2)
        target_exp_mean, target_exp_logvar = model.exp_encoder(target_exp)
        exp_mean, exp_logvar = model.exp_encoder(pred_exp)
        # sampling
        if prob_method == 'csd' :
            mu_pdist = ((target_exp_mean - exp_mean) ** 2).sum(-1)
            sigma_pdist = ((torch.exp(target_exp_logvar) + torch.exp(exp_logvar))).sum(-1)
            logits = mu_pdist + sigma_pdist
            logits = -model.criterion.negative_scale * logits + model.criterion.shift # (Bs, BS)
            labels = torch.ones(logits.shape[0], dtype=logits.dtype, device=logits.device)
            loss = model.criterion.bceloss(logits, labels)
            return loss
        elif prob_method == 'mean':
            mu_pdist = ((target_exp_mean - exp_mean) ** 2).sum(-1)
            return mu_pdist.mean()
        
        elif prob_method == 'sample' :
            raise NotImplementedError
            DEE_target_exp_embedding = sample_gaussian_tensors(target_exp_mean, target_exp_logvar, num_samples, normalize=True)
            DEE_exp_embedding = sample_gaussian_tensors(exp_mean, exp_logvar, num_samples, normalize=True)
            if num_samples == 1 :
                DEE_audio_embedding = DEE_target_exp_embedding.squeeze(1)
                DEE_exp_embedding = DEE_exp_embedding.squeeze(1)
    raise ValueError(f"Invalid prob_method '{prob_method}'")
