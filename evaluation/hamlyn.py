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
                  device: Device = 'cpu',
                  no_pbar: bool = False) -> float:

    model.eval()

    running_left_ssim = 0
    running_right_ssim = 0

    ssims = []
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

        left_score, left_ssim = u.calculate_ssim(left, left_recon)
        right_score, right_ssim = u.calculate_ssim(right, right_recon)

        ssims.append((left_score, right_score))

        left_l1 = (left - left_recon).abs()
        right_l1 = (right - right_recon).abs()

        left_ssim = torch.tensor(left_ssim).to(device)
        right_ssim = torch.tensor(right_ssim).to(device)

        left_error = ((ssim_weight * left_ssim) \
            + (1 - ssim_weight) * left_l1)
        right_error = ((ssim_weight * right_ssim) \
            + (1 - ssim_weight) * right_l1)

        true_error = torch.cat((left_error, right_error), dim=1)

        spars_curve = s.sparsification_curve(true_error, pred_error)
        oracle_curve = s.sparsification_curve(true_error, true_error)
        random_curve = s.random_sparsification_curve(true_error)

        spars_curves.append((spars_curve, oracle_curve, random_curve))

        average_left_ssim = running_left_ssim / ((i+1) * batch_size)
        average_right_ssim = running_right_ssim / ((i+1) * batch_size)

        tepoch.set_postfix(left=average_left_ssim,
                           right=average_right_ssim)

        if save_results_to is not None:
            differences = torch.cat((left, right, 
                                    left_disp, right_disp, 
                                    left_recon, right_recon, 
                                    left_ssim, right_ssim), 0)

            differences_image = make_grid(differences, nrow=2)
            filepath = os.path.join(save_results_to, f'image_{i:04}.png')

            save_image(differences_image, filepath)

    if no_pbar:
        print(f'{description}:'
              f'\n\tAverage left SSIM score: {left_score:.3f}'
              f'\n\tAverage right SSIM score: {right_score:.3f}')

    return ssims, spars_curves
