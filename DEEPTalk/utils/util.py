import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as t
import torchvision.transforms.functional as TF
import face_alignment
from scipy.signal import find_peaks
import numpy as np


def sample_gaussian_tensors(mu, logsigma, num_samples, normalize):
    if num_samples == 0:
        return mu.unsqueeze(1)
    logsigma = logsigma * 0.5
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), # BS, num_samples, dim
                    dtype=mu.dtype, device=mu.device)

    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
        mu.unsqueeze(1))
    if normalize:
        return F.normalize(samples, p=2, dim=-1)
    else:
        return samples

def detect_landmarks(landmarks_detector, images, device) :
    transform = t.Compose([t.Resize((224,224))])

    if images.shape[3] > 224 :
        images = transform(images)
    else :
        images = images
    images = images*255.0
    BS,T,C,H,W = images.shape
    images = images.reshape(BS*T,C,H,W)


    landmarks = landmarks_detector.get_landmarks_from_batch(images.to(device, dtype=torch.float32))
    landmarks = np.array(landmarks)
    landmarks = torch.tensor(landmarks).to(device)
    landmarks = landmarks.reshape(BS,T,68,2)

    return landmarks

def cut_mouth_vectorized(images, landmarks, device,
                        mouth_crop_width = 96,
                        mouth_crop_height = 96,
                        mouth_window_margin = 12,
                        mouth_landmark_start_idx = 48,
                        mouth_landmark_stop_idx = 68,
                        convert_grayscale = True) :
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
        temporal_filter = torch.ones(mouth_window_margin, device=images.device) / mouth_window_margin
        # pad the the landmarks 
        landmarks_t_padded = F.pad(landmarks_t, (mouth_window_margin // 2, mouth_window_margin // 2), mode='replicate')
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
        mouth_landmarks_t = smooth_landmarks_t[..., mouth_landmark_start_idx:mouth_landmark_stop_idx, :]
        # print(f'mouth_landmark : {mouth_landmarks_t}')
        
        # #3) get the mean of the mouth landmarks
        mouth_landmarks_mean_t = mouth_landmarks_t.mean(dim=-2, keepdims=True)
    
        # #4) get the center of the mouth
        center_x_t = mouth_landmarks_mean_t[..., 0]
        center_y_t = mouth_landmarks_mean_t[..., 1]

        # #5) use grid_sample to crop the mouth in every image 
        # create the grid
        height = mouth_crop_height//2
        width = mouth_crop_width//2

        torch.arange(0, mouth_crop_width, device=images.device)

        grid = torch.stack(torch.meshgrid(torch.linspace(-height, height, mouth_crop_height).to(images.device) / (images.shape[-2] /2),
                                torch.linspace(-width, width, mouth_crop_width).to(images.device) / (images.shape[-1] /2) ), 
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

    if convert_grayscale: 
        images = TF.rgb_to_grayscale(images)

    image_crops = F.grid_sample(
        images, 
        grid,  
        align_corners=True, 
        padding_mode='zeros',
        mode='bicubic'
        )
    image_crops = image_crops.view(B, T, *image_crops.shape[1:])

    if convert_grayscale:
        image_crops = image_crops#.squeeze(1)

    return image_crops