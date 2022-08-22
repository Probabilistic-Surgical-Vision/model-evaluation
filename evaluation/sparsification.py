from typing import List

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

DataPoints = List[float]


def curve(oracle_error: Tensor, predicted_error: Tensor,
          kernel_size: int = 11, steps: int = 100) -> DataPoints:
    
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
        normalised_mean = (slice_mean / oracle_mean).mean().item()

        curve.append(normalised_mean)

    return np.array(curve)


def random_curve(oracle_error: Tensor, kernel_size: int = 11) -> DataPoints:
    random_error = torch.rand_like(oracle_error)
    return curve(oracle_error, random_error, kernel_size)


def error(oracle_curve: DataPoints,
          predicted_curve: DataPoints) -> DataPoints:

    return predicted_curve - oracle_curve


def ause(oracle_curve: DataPoints, predicted_curve: DataPoints) -> float:
    if len(oracle_curve) != len(predicted_curve):
        raise Exception('Oracle and Predicted sparsification '
                        'curves have different step sizes.')

    dx = 1 / len(oracle_curve)
    integral = 0

    for oracle_error, predicted_error in zip(oracle_curve, predicted_curve):
        integral += (predicted_error - oracle_error) * dx

    return integral


def aurg(predicted_curve: DataPoints, random_curve: DataPoints) -> float:
    return ause(predicted_curve, random_curve)
