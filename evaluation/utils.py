from typing import OrderedDict, Union

import matplotlib.pyplot as plt

import numpy as np

from skimage.metrics import structural_similarity

import torch
import torch.nn.functional as F
from torch import Tensor

Device = Union[str, torch.device]


def disparity_to_depth(disparity: Tensor, focal_length: Tensor,
                       baseline: float) -> Tensor:
    # Note baseline is given in mm, so divide by 1000
    focal_length = focal_length.reshape(-1, 1, 1, 1)
    baseline = baseline.reshape(-1, 1, 1, 1)
    return focal_length * (baseline / 1000) / disparity


def postprocess_disparity(left: Tensor, right: Tensor, device: Device = 'cpu',
                          alpha: float = 20, beta: float = 0.05) -> Tensor:

    left_disp = left.cpu().numpy()
    right_disp = right.cpu().numpy()
    mean_disp = (left_disp + right_disp) / 2

    _, _, height, width = mean_disp.shape

    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, _ = np.meshgrid(x, y)

    left_mask = 1 - np.clip(alpha * (xv - beta), 0, 1)
    left_mask = np.expand_dims(left_mask, (0, 1))

    right_mask = np.flip(left_mask, axis=3)

    mean_mask = 1 - (left_mask + right_mask)

    combined_disparity = (right_mask * left_disp) \
        + (left_mask * right_disp) \
        + (mean_mask * mean_disp)

    return torch.from_numpy(combined_disparity).to(device)


def mean_absolute_depth(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).abs().mean()


def reconstruct(disparity: Tensor, opposite_image: Tensor) -> Tensor:
    batch_size, _, height, width = opposite_image.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width) \
        .repeat(batch_size, height, 1) \
        .type_as(opposite_image)

    y_base = torch.linspace(0, 1, height) \
        .repeat(batch_size, width, 1) \
        .transpose(1, 2) \
        .type_as(opposite_image)

    # Apply shift in X direction
    x_shifts = disparity.squeeze(dim=1)

    # In grid_sample coordinates are assumed to be between -1 and 1
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    flow_field = (2 * flow_field) - 1

    return F.grid_sample(opposite_image, flow_field, mode='bilinear',
                         padding_mode='zeros')


def reconstruct_left_image(left_disparity: Tensor,
                           right_image: Tensor) -> Tensor:

    return reconstruct(-left_disparity, right_image)


def reconstruct_right_image(right_disparity: Tensor,
                            left_image: Tensor) -> Tensor:

    return reconstruct(right_disparity, left_image)


def calculate_ssim(a: Tensor, b: Tensor, window_size: int = 11, device: Device = 'cpu') -> Tensor:
    a_numpy = a.cpu().numpy()
    b_numpy = b.cpu().numpy()

    score, diff = structural_similarity(a_numpy, b_numpy, win_size=window_size,
                                        channel_axis=0,
                                        full=True)
    
    return score, torch.from_numpy(diff).to(device)

def prepare_state_dict(state_dict: OrderedDict) -> dict:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def to_heatmap(x: Tensor, scale: bool = True, inverse: bool = False,
               colour_map: str = 'inferno', device: Device = 'cpu') -> Tensor:

    if scale:
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

    image = x.squeeze(0).cpu().numpy()
    image = 1 - image if inverse else image

    transform = plt.get_cmap(colour_map)
    heatmap = transform(image)[:, :, :3]  # remove alpha channel

    return torch.from_numpy(heatmap).to(device).permute(2, 0, 1)

