from typing import Union

import torch
from torch import Tensor

Device = Union[str, torch.device]


def disparity_to_depth(disparity: Tensor, focal_length: float,
                       baseline: float) -> Tensor:
    # Note baseline is given in mm, so divide by 1000
    return focal_length * (baseline / 1000) / disparity


def convert_scared_video_to_images():
    pass


def postprocess_disparity(disparity: Tensor) -> Tensor:
    pass


def mean_absolute_depth(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).abs().mean()
