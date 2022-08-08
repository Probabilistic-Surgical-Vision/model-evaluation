import glob
import os.path
from typing import Optional

import numpy as np
from numpy import ndarray

from PIL import Image, ImageFile

import tifffile

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import tqdm

import yaml
from yaml import Loader, Node

from . import utils as u
from .utils import Device

ImageFile.LOAD_TRUNCATED_IMAGES = True


def opencv_matrix(loader: Loader, node: Node) -> ndarray:
    mapping = loader.construct_mapping(node, deep=True)
    return np.array(mapping['data']) \
        .reshape((mapping['rows'], mapping['cols']))


yaml.add_constructor(u'tag:yaml.org,2002:opencv-matrix', opencv_matrix)


class SCAREDKeyframesLoader(Dataset):

    KEYframeS_FOLDER_GLOB = 'keyframe_*'
    LEFT_IMAGE = 'Left_Image.png'
    RIGHT_IMAGE = 'Right_Image.png'
    LEFT_DEPTH = 'left_depth_map.tiff'
    RIGHT_DEPTH = 'right_depth_map.tiff'
    CALIBRATION = 'endoscope_calibration.yml'
    DEPTH_SIZE = (1024, 1280)

    def __init__(self, root: str, split: str,
                 dataset_number: Optional[int] = None,
                 image_transform: Optional[object] = None,
                 depth_transform: Optional[object] = None) -> None:

        if split not in ('train', 'test'):
            raise ValueError('Split must be either "train" or "test".')

        dataset_path = '**' if dataset_number is None \
            else f'dataset_{dataset_number}'

        keyframes_glob = os.path.join(root, split, dataset_path,
                                      self.KEYframeS_FOLDER_GLOB)

        self.keyframes = sorted(glob.glob(keyframes_glob))

        self.image_transform = image_transform
        self.depth_transform = depth_transform

    def get_x_focal_length(self, intrinsic: ndarray) -> Tensor:
        return torch.tensor(intrinsic[0, 0])

    def get_x_baseline(self, translation: ndarray) -> Tensor:
        return torch.tensor([translation[0]])

    def __getitem__(self, idx: int) -> dict:
        keyframes_folder = self.keyframes[idx]

        left_image_path = os.path.join(keyframes_folder, self.LEFT_IMAGE)
        right_image_path = os.path.join(keyframes_folder, self.RIGHT_IMAGE)
        left_depth_path = os.path.join(keyframes_folder, self.LEFT_DEPTH)
        right_depth_path = os.path.join(keyframes_folder, self.RIGHT_DEPTH)

        calibration_path = os.path.join(keyframes_folder, self.CALIBRATION)

        left_image = Image.open(left_image_path).convert('RGB')
        right_image = Image.open(right_image_path).convert('RGB')

        left_depth = tifffile.imread(left_depth_path)
        right_depth = tifffile.imread(right_depth_path)

        with open(calibration_path) as f:
            calibration = yaml.load(f, loader=yaml.Loader)

        image_pair = {'left': left_image, 'right': right_image}
        depth_pair = {'left': left_depth, 'right': right_depth}

        if self.image_transform is not None:
            image_pair = self.image_transform(image_pair)
        if self.depth_transform is not None:
            image_pair = self.depth_transform(depth_pair)

        _, height, width = left_depth.size()

        if (height, width) != self.DEPTH_SIZE:
            raise Exception(f'Depth maps must have size {self.DEPTH_SIZE}'
                            ' to accurately evaluate the model on the'
                            ' SCARED Dataset')

        return {
            'images': image_pair,
            'depth': depth_pair,
            'focal': self.get_x_focal_length(calibration['M1']),
            'baseline': self.get_x_baseline(calibration['T'])
        }

    def __len__(self) -> int:
        return len(self.keyframes)


def evaluate_keyframes(model: Module, loader: DataLoader,
                       min_depth: float, max_depth: float,
                       device: Device = 'cpu', no_pbar: bool = False):

    model.eval()
    results = []

    tepoch = tqdm.tqdm(loader, unit='batch', disable=no_pbar)

    for keyframe in tepoch:
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

        tepoch.set_postfix(mae=mae)
        results.append(mae)

    return results
