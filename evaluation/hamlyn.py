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
    running_ause = 0
    running_aurg = 0

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

        left_score = left_scores.mean().item()
        right_score = right_scores.mean().item()

        ssim_scores.append((left_score, right_score))

        running_left_score += left_score
        running_right_score += right_score

        average_left_ssim = running_left_score / (i+1)
        average_right_ssim = running_right_score / (i+1)

        if prediction.size(1) == 4:
            uncertainty = prediction[:, 2:]

            left_l1 = (left - left_recon).abs()
            right_l1 = (right - right_recon).abs()

            l1 = torch.cat((left_l1, right_l1), dim=1)

            weight_tensor = torch.full_like(ssims, ssim_weight)

            true_error = (weight_tensor * ((1 - ssims) / 2).abs()) \
                + ((1 - weight_tensor) * l1)

            left_error, right_error = torch.split(true_error, [3, 3], dim=1)
            true_error = torch.cat((left_error.mean(1, True),
                                   right_error.mean(1, True)), dim=1)

            oracle = spars.curve(true_error, true_error,
                                 kernel, device=device)
            pred_spars = spars.curve(true_error, uncertainty,
                                     kernel, device=device)
            random_spars = spars.random_curve(true_error, kernel,
                                              device=device)

            ause = spars.ause(oracle, pred_spars)
            aurg = spars.aurg(pred_spars, random_spars)

            running_ause += ause.item()
            average_ause = running_ause / (i+1)

            running_aurg += aurg.item()
            average_aurg = running_aurg / (i+1)

            oracle = oracle.cpu()
            pred_spars = pred_spars.cpu()
            random_spars = random_spars.cpu()

            spars_curves.append((pred_spars, oracle, random_spars))

            tepoch.set_postfix(left=average_left_ssim,
                               right=average_right_ssim,
                               ause=average_ause,
                               aurg=average_aurg)

        else:
            tepoch.set_postfix(left=average_left_ssim,
                               right=average_right_ssim)

        if save_results_to is not None and i % save_every == 0:
            batch_save_to = os.path.join(save_results_to, f'batch_{i:04}')
            
            if not os.path.isdir(batch_save_to):
                os.makedirs(batch_save_to, exist_ok=True)
            
            left_path = os.path.join(batch_save_to, 'image.png')
            save_image(left[0], left_path)
            
            left_disp_image = u.to_heatmap(left_disp[0], device=device)
            left_disp_path = os.path.join(batch_save_to, 'disparity.png')
            save_image(left_disp_image, left_disp_path)
            
            left_recon_path = os.path.join(batch_save_to, 'reconstruction.png')
            save_image(left_recon[0], left_recon_path)
            
            left_ssim_path = os.path.join(batch_save_to, 'ssim.png')
            save_image(ssims[0, 0:3], left_ssim_path)
            
            if prediction.size(1) == 4:
                left_uncertainty_image = u.to_heatmap(uncertainty[0, 0:1], device=device)
                left_uncertainty_path = os.path.join(batch_save_to, 'uncertainty.png')
                save_image(left_uncertainty_image, left_uncertainty_path)
                
                left_error_image = u.to_heatmap(true_error[0, 0:1], device=device)
                left_error_path = os.path.join(batch_save_to, 'error.png')
                save_image(left_error_image, left_error_path)

                single_row_images = torch.stack((left[0], left_disp_image,
                                                 left_uncertainty_image,
                                                 left_error_image))
            else:
                single_row_images = torch.stack((left[0], left_disp_image,
                                                 ssims[0, 0:3]))
        
            single_row_image = make_grid(single_row_images, nrow=4)

            single_row_path = os.path.join(save_results_to, 'single_rows')
            filepath = os.path.join(single_row_path, f'image_{i:04}.png')

            if not os.path.isdir(single_row_path):
                os.makedirs(single_row_path, exist_ok=True)

            save_image(single_row_image, filepath)
            
            """
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
            
            single_row_images = torch.stack((left[0], left_disp,
                                             uncertainty, true_error))
        
            single_row_image = make_grid(single_row_images, nrow=4)

            single_row_path = os.path.join(save_results_to, 'single_rows')
            filepath = os.path.join(single_row_path, f'image_{i:04}.png')

            if not os.path.isdir(single_row_path):
                os.makedirs(single_row_path, exist_ok=True)

            save_image(single_row_image, filepath)
            """

    if no_pbar:
        print(f'{description}:'
              f'\n\tAverage left SSIM score: {average_left_ssim:.3f}'
              f'\n\tAverage right SSIM score: {average_right_ssim:.3f}')

    return ssim_scores, spars_curves
