import os.path
from typing import Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from torchvision.utils import make_grid, save_image

import tqdm

from . import utils as u
from .utils import Device


@torch.no_grad()
def evaluate_ssim(model: Module, loader: DataLoader,
                  save_results_to: Optional[str] = None,
                  device: Device = 'cpu', no_pbar: bool = False) -> float:

    model.eval()

    running_left_ssim = 0
    running_right_ssim = 0

    ssims = []

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
        else len(loader)

    description = 'SSIM Evaluation'
    tepoch = tqdm.tqdm(loader, description, unit='batch', disable=no_pbar)

    for i, image_pair in enumerate(tepoch):
        left = image_pair['left'].to(device)
        right = image_pair['right'].to(device)

        prediction = model(left)

        left_disp, right_disp = torch.split(prediction[:, :2], [1, 1], 1)

        left_recon = u.reconstruct_left_image(left_disp, right)
        right_recon = u.reconstruct_right_image(right_disp, left)

        left_ssim, left_diff = u.calculate_ssim(left, left_recon)
        right_ssim, right_diff = u.calculate_ssim(right, right_recon)

        ssims.append((left_ssim, right_ssim))

        average_left_ssim = running_left_ssim / ((i+1) * batch_size)
        average_right_ssim = running_right_ssim / ((i+1) * batch_size)

        tepoch.set_postfix(left=average_left_ssim,
                           right=average_right_ssim)

        if save_results_to is not None:
            differences = torch.cat((left_diff, right_diff), 0)
            differences_image = make_grid(differences, nrow=2)
            filepath = os.path.join(save_results_to, f'image_{i:04}.png')

            save_image(differences_image, filepath)

    if no_pbar:
        print(f'{description}:'
              f'\n\tAverage left SSIM score: {left_ssim:.3f}'
              f'\n\tAverage right SSIM score: {right_ssim:.3f}')

    return ssims
