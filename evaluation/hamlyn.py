import os
import os.path
from typing import Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

from torchmetrics.functional import \
    structural_similarity_index_measure as ssim

import tqdm

from . import sparsification as spars

from . import utils as u
from .utils import Device


@torch.no_grad()
def evaluate_ssim(model: Module, loader: DataLoader,
                  save_results_to: Optional[str] = None,
                  ssim_weight: float = 0.85,
                  kernel: int = 11,
                  save_every: int = 50,
                  device: Device = 'cpu',
                  no_pbar: bool = False) -> float:

    model.eval()

    running_left_score = 0
    running_right_score = 0

    ssim_scores = []
    spars_curves = []

    description = 'SSIM Evaluation'
    tepoch = tqdm.tqdm(loader, description, unit='batch', disable=no_pbar)

    for i, image_pair in enumerate(tepoch):
        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

        prediction = model(left)

        disparity = prediction[:, :2]
        left_disp, right_disp = torch.split(disparity, [1, 1], 1)

        left_recon = u.reconstruct_left_image(left_disp, right)
        right_recon = u.reconstruct_right_image(right_disp, left)

        left_scores, left_ssims = ssim(left_recon, left,
                                       kernel_size=kernel,
                                       return_full_image=True,
                                       reduction='none')
        right_scores, right_ssims = ssim(right_recon, right,
                                         kernel_size=kernel,
                                         return_full_image=True,
                                         reduction='none')

        ssims = torch.cat((left_ssims, right_ssims), dim=1)

        running_left_score += left_scores.mean().item()
        running_right_score += right_scores.mean().item()

        average_left_ssim = running_left_score / (i+1)
        average_right_ssim = running_right_score / (i+1)

        tepoch.set_postfix(left=average_left_ssim,
                           right=average_right_ssim)

        if prediction.size(1) == 4:
            uncertainty = prediction[:, 2:]

            left_l1 = (left - left_recon).abs()
            right_l1 = (right - right_recon).abs()

            l1 = torch.cat((left_l1, right_l1), dim=1)

            weight_tensor = torch.full_like(ssims, ssim_weight)

            true_error = (weight_tensor * ((1 - ssims) / 2).abs()) \
                + ((1 - weight_tensor) * l1)

            left_error, right_error = torch.split(true_error, [3, 3], dim=1)
            true_error = torch.cat((right_error.mean(1, True),
                                   left_error.mean(1, True)), dim=1)

            oracle = spars.curve(true_error, true_error,
                                 kernel, device=device)
            pred_curve = spars.curve(true_error, uncertainty,
                                     kernel, device=device)
            random_curve = spars.random_curve(true_error, kernel,
                                              device=device)

            oracle = oracle.cpu()
            pred_curve = pred_curve.cpu()
            random_curve = random_curve.cpu()

            spars_curves.append((pred_curve, oracle, random_curve))

        if save_results_to is not None and i % save_every == 0:
            left_disp = u.to_heatmap(left_disp[0], device=device)
            disparity = torch.stack((left[0], left_disp,
                                    left_recon[0], ssims[0, 0:3]))

            if prediction.size(1) == 4:
                uncertainty = u.to_heatmap(uncertainty[0, 0:1], device=device)
                true_error = u.to_heatmap(true_error[0, 0:1], device=device)

                error = torch.stack((uncertainty, true_error))
                disparity = torch.cat((disparity, error))

            disparity_image = make_grid(disparity, nrow=2)
            filepath = os.path.join(save_results_to, f'image_{i:04}.png')

            if not os.path.isdir(save_results_to):
                os.makedirs(save_results_to, exist_ok=True)

            save_image(disparity_image, filepath)

    if no_pbar:
        print(f'{description}:'
              f'\n\tAverage left SSIM score: {average_left_ssim:.3f}'
              f'\n\tAverage right SSIM score: {average_right_ssim:.3f}')

    return ssim_scores, spars_curves
