import os
import os.path
from typing import Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

import tqdm

from . import sparsification as s

from . import utils as u
from .utils import Device


@torch.no_grad()
def evaluate_ssim(model: Module, loader: DataLoader,
                  save_results_to: Optional[str] = None,
                  ssim_weight: float = 0.85,
                  save_every: int = 50,
                  device: Device = 'cpu',
                  no_pbar: bool = False) -> float:

    model.eval()

    running_left_score = 0
    running_right_score = 0

    ssim_scores = []
    spars_curves = []

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
        else len(loader)

    description = 'SSIM Evaluation'
    tepoch = tqdm.tqdm(loader, description, unit='batch', disable=no_pbar)

    for i, image_pair in enumerate(tepoch):
        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

        prediction = model(left)
        pred_disp, pred_error = torch.split(prediction, [2, 2], dim=1)
        left_disp, right_disp = torch.split(pred_disp, [1, 1], 1)

        left_recon = u.reconstruct_left_image(left_disp, right)
        right_recon = u.reconstruct_right_image(right_disp, left)

        ssims = []

        # Can't batch process SSIM so they are done individually
        for j in range(batch_size):
            left_score, left_ssim = u.calculate_ssim(left[j], left_recon[j],
                                                     device=device)
            right_score, right_ssim = u.calculate_ssim(right[j], right_recon[j],
                                                       device=device)

            running_left_score += left_score
            running_right_score += right_score

            ssim_scores.append((left_score, right_score))

            ssims.append(torch.cat((left_ssim, right_ssim), dim=0))

        ssim = torch.stack(ssims, dim=0)

        left_l1 = (left - left_recon).abs()
        right_l1 = (right - right_recon).abs()

        l1 = torch.cat((left_l1, right_l1), dim=1)

        weight_tensor = torch.full_like(ssim, ssim_weight)

        true_error = (weight_tensor * ssim) + ((1 - weight_tensor) * l1)

        left_error, right_error = torch.split(true_error, [3, 3], dim=1)
        left_error = left_error.mean(1, True)
        right_error = right_error.mean(1, True)

        true_error = torch.cat((left_error, right_error), dim=1)

        spars_curve = s.sparsification_curve(true_error, pred_error)
        oracle_curve = s.sparsification_curve(true_error, true_error)
        random_curve = s.random_sparsification_curve(true_error)

        spars_curves.append((spars_curve, oracle_curve, random_curve))

        average_left_ssim = running_left_score / ((i+1) * batch_size)
        average_right_ssim = running_right_score / ((i+1) * batch_size)

        tepoch.set_postfix(left=average_left_ssim,
                           right=average_right_ssim)

        if save_results_to is not None and i % save_every == 0:
            left_ssim, right_ssim = torch.split(ssim, [3, 3], dim=1)

            min_disp, max_disp = left_disp[0].min(), left_disp[0].max()
            left_disp_scaled = (left_disp[0] - min_disp) / (max_disp - min_disp)

            left_disp_heat = u.to_heatmap(left_disp_scaled, device)

            disparity = torch.stack((left[0], left_disp_heat,
                                    left_recon[0], left_ssim[0]))

            disparity_image = make_grid(disparity, nrow=2)
            filepath = os.path.join(save_results_to, f'image_{i:04}.png')

            if not os.path.isdir(save_results_to):
                os.makedirs(save_results_to, exist_ok=True)

            save_image(disparity_image, filepath)

    if no_pbar:
        print(f'{description}:'
              f'\n\tAverage left SSIM score: {left_score:.3f}'
              f'\n\tAverage right SSIM score: {right_score:.3f}')

    return ssim_scores, spars_curves
