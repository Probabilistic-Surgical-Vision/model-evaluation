import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader

import tqdm

from . import utils as u
from .utils import Device


@torch.no_grad()
def evaluate_keyframes(model: Module, loader: DataLoader,
                       min_depth: float, max_depth: float,
                       device: Device = 'cpu', no_pbar: bool = False):
    model.eval()

    running_mae = 0
    results = []

    batch_size = loader.batch_size \
        if loader.batch_size is not None \
        else len(loader)

    tepoch = tqdm.tqdm(loader, unit='batch', disable=no_pbar)

    for i, keyframe in enumerate(tepoch):
        images = keyframe['images']
        depth = keyframe['depth']

        left_image = images['left'].to(device)
        left_depth = depth['left'].to(device)

        _, _, height, width = left_depth.size()

        predictions = model(left_image)

        disparity = u.postprocess_disparity(predictions[:, :3])
        disparity = F.interpolate(disparity, (height, width),
                                  mode='bilinear', align_corners=True)

        f, b = keyframe['focal'], keyframe['baseline']
        predicted_depth = u.disparity_to_depth(disparity, f, b)
        predicted_depth = torch.clip(predicted_depth, min_depth, max_depth)

        predicted_depth = torch.cat([predicted_depth] * 3, dim=1)

        if predicted_depth.isnan().any():
            raise Exception('Predicted depth map contains NaNs.')
        elif left_depth.isnan().any():
            raise Exception('Left depth map contains NaNs.')

        mask = left_depth.gt(0)

        masked_predicted_depth = predicted_depth.masked_select(mask)
        masked_left_depth = left_depth.masked_select(mask)

        mae = u.mean_absolute_depth(masked_predicted_depth, masked_left_depth)

        results.append(mae)

        running_mae += mae
        average_mae = running_mae / ((i+1) * batch_size)

        tepoch.set_postfix(mae=average_mae)

    return results
