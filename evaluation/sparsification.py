import torch
import torch.nn as nn
from torch import Tensor

from .utils import Device


def curve(oracle_error: Tensor, predicted_error: Tensor,
          kernel_size: int = 11, steps: int = 100,
          device: Device = 'cpu') -> Tensor:

    batch_size = predicted_error.size(0)
    pool = nn.AvgPool2d(kernel_size, stride=1)

    oracle_error = pool(oracle_error).view(batch_size, 2, -1)
    predicted_error = pool(predicted_error).view(batch_size, 2, -1)

    predicted_indices = predicted_error.argsort(2, True)
    oracle_sorted_by_error = oracle_error.gather(2, predicted_indices)

    oracle_mean = oracle_error.mean(dim=2)

    curve = []

    for step in range(steps):
        fraction = step / steps
        removed_pixels = int(fraction * oracle_error.size(2))

        slice = oracle_sorted_by_error[:, :, removed_pixels:]

        slice_mean = slice.mean(dim=2)
        normalised_mean = (slice_mean / oracle_mean).mean()

        curve.append(normalised_mean)

    return torch.tensor(curve, device=device)


def random_curve(oracle_error: Tensor, kernel_size: int = 11,
                 steps: int = 100, device: Device = 'cpu') -> Tensor:

    random_error = torch.rand_like(oracle_error)
    return curve(oracle_error, random_error, kernel_size, steps, device)


def error(oracle_curve: Tensor,
          predicted_curve: Tensor) -> Tensor:

    return predicted_curve - oracle_curve


def ause(oracle_curve: Tensor, predicted_curve: Tensor) -> Tensor:
    if len(oracle_curve) != len(predicted_curve):
        raise Exception('Oracle and Predicted sparsification '
                        'curves have different step sizes.')

    return error(oracle_curve, predicted_curve).sum() / len(oracle_curve)


def aurg(predicted_curve: Tensor, random_curve: Tensor) -> Tensor:
    return ause(predicted_curve, random_curve)
