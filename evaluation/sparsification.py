from typing import List

import torch
from torch import Tensor

DataPoints = List[float]


def sparsification_curve(oracle_error: Tensor, predicted_error: Tensor,
                         steps: int = 100) -> DataPoints:

    batch_size = predicted_error.size(0)

    oracle_error = oracle_error.view(batch_size, -1)
    predicted_error = predicted_error.view(batch_size, -1)

    predicted_indices = predicted_error.argsort(1, True)
    oracle_sorted_by_error = oracle_error.gather(1, predicted_indices)

    oracle_mean = oracle_error.mean(dim=1)

    curve = []

    for step in range(steps):
        fraction = step / steps
        removed_pixels = int(fraction * oracle_error.size(1))

        slice = oracle_sorted_by_error[:, removed_pixels:]

        mean = slice.mean(dim=1)
        normalised_mean = float((mean / oracle_mean).mean())

        curve.append(normalised_mean)

    return curve


def sparsification_error(oracle_curve: DataPoints,
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


def random_sparsification_curve(oracle_error: Tensor):
    random_error = torch.rand_like(oracle_error)
    return sparsification_curve(oracle_error, random_error)


def aurg(predicted_curve: DataPoints, random_curve: DataPoints) -> float:
    return ause(predicted_curve, random_curve)
