import os
from typing import Dict, Optional

import numpy as np
from numpy import float32, ndarray

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

import tqdm

from . import utils as u
from .utils import Device


@torch.no_grad()
def evaluate_keyframes(model: Module, loader: DataLoader,
                       min_depth: float, max_depth: float,
                       camera_parameters: Optional[Dict[str, float]] = None,
                       save_results_to: Optional[str] = None,
                       device: Device = 'cpu',
                       no_pbar: bool = False):
    model.eval()

    running_mae = 0
    
    metrics = []
    maes = []

    if loader.batch_size != 1:
        raise Exception('Keyframes evaluation must have a batch size of 1.')

    tepoch = tqdm.tqdm(loader, unit='batch', disable=no_pbar)

    for i, keyframe in enumerate(tepoch):
        images = keyframe['images']
        depth = keyframe['depth']
        
        if camera_parameters is not None:
            focal = camera_parameters['focal_length']
            base = camera_parameters['baseline']
            
            f = torch.tensor([focal]).to(device)
            b = torch.tensor([base]).to(device)
        else:
            f = keyframe['focal'].to(device)
            b = keyframe['baseline'].to(device)

        left = images['left'].to(device)
        right = images['right'].to(device)

        depth = depth['left'].to(device)

        _, _, height, width = depth.size()

        prediction = model(left)
        prediction = F.interpolate(prediction, size=(height, width),
                                   mode='bilinear', align_corners=True)

        left_disp, right_disp = torch.split(prediction[:, :2], [1, 1], dim=1)
        disparity = u.postprocess_disparity(left_disp, right_disp, device)

        f, b = keyframe['focal'].to(device), keyframe['baseline'].to(device)
        pred_depth = u.disparity_to_depth(disparity, f, b)
        pred_depth = torch.clip(pred_depth, min_depth, max_depth)
        
        pred_depth = pred_depth.cpu().numpy()
        depth = depth.cpu().numpy()
        depth = np.nan_to_num(depth)
        
        mask = depth < 0.1
        
        difference = np.abs(pred_depth - depth)
        error_image = np.ma.array(difference, mask=mask)
        mean_error = error_image.mean()

        mask = np.logical_and(depth > min_depth,
                              depth < max_depth)
        
        keyframe_metrics = error_metrics(pred_depth[mask],
                                         depth[mask])

        maes.append(mean_error)
        metrics.append(keyframe_metrics)

        running_mae += mean_error
        average_mae = running_mae / (i+1)

        if save_results_to is not None:
            pred_depth = torch.from_numpy(pred_depth[0]).to(device)
            pred_depth = u.to_heatmap(pred_depth, device=device)

            true_depth = torch.from_numpy(depth[0]).to(device)
            true_depth = u.to_heatmap(true_depth, device=device)

            left_recon = u.reconstruct_left_image(left_disp, right)

            difference[mask] = 0
            difference = torch.from_numpy(difference).to(device)

            disparity = torch.stack((left[0], pred_depth,
                                    left_recon[0], true_depth))

            if prediction.size(1) == 4:
                uncertainty = u.to_heatmap(uncertainty[0, 0:1], device=device)

                difference[mask] = 0
                difference = torch.from_numpy(difference).to(device)

                error = torch.stack((uncertainty, difference))
                disparity = torch.cat((disparity, error))

            disparity_image = make_grid(disparity, nrow=2)
            filepath = os.path.join(save_results_to, f'image_{i:04}.png')

            if not os.path.isdir(save_results_to):
                os.makedirs(save_results_to, exist_ok=True)

            save_image(disparity_image, filepath)

        tepoch.set_postfix(mae=average_mae)

    return maes, metrics


def error_metrics(prediction: ndarray, truth: ndarray) -> Dict[str, float32]:
    threshold = np.maximum((truth / prediction), (prediction / truth))

    difference: ndarray = truth - prediction
    square_difference = (prediction - truth) ** 2
    square_log_difference = (np.log(truth) - np.log(prediction)) ** 2

    return {
        'a1': (threshold < 1.25).mean(),
        'a2': (threshold < 1.25 ** 2).mean(),
        'a3': (threshold < 1.25 ** 3).mean(),
        'rmse': np.sqrt(square_difference.mean()),
        'rmse_log': np.sqrt(square_log_difference.mean()),
        'abs_rel': (np.abs(difference) / truth).mean(),
        'square_rel': ((difference ** 2) / truth).mean(),
        'mean_abs': np.abs(difference).mean()
    }