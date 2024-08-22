import numpy as np
import torch

def vertex_error(pred, gt) :
    """
    Calculate the vertex error between predicted and ground truth values.
    NOTE : used as LVE(lip vertex error) and EVE(EMotion vertex error) in Emotalk
    Args:
        pred (torch.Tensor): Predicted values.
        gt (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Vertex error.
    """
    return torch.mean(torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1)))