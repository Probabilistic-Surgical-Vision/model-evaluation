import glob
import os.path
from typing import Optional

from PIL import Image, ImageFile

import tifffile

import torch
from torch.utils.data import Dataset

import yaml

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SCAREDKeyframesLoader(Dataset):

    KEYFRAMES_FOLDER_GLOB = 'keyframe_*'
    LEFT_IMAGE = 'Left_Image.png'
    RIGHT_IMAGE = 'Right_Image.png'
    LEFT_DEPTH = 'left_depth_map.tiff'
    RIGHT_DEPTH = 'right_depth_map.tiff'
    CAMERA_PARAMETERS = 'camera_parameters.yml'
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
                                      self.KEYFRAMES_FOLDER_GLOB)

        self.keyframes = sorted(glob.glob(keyframes_glob))

        self.image_transform = image_transform
        self.depth_transform = depth_transform

    def __getitem__(self, idx: int) -> dict:
        keyframes_folder = self.keyframes[idx]

        left_image_path = os.path.join(keyframes_folder, self.LEFT_IMAGE)
        right_image_path = os.path.join(keyframes_folder, self.RIGHT_IMAGE)
        left_depth_path = os.path.join(keyframes_folder, self.LEFT_DEPTH)
        right_depth_path = os.path.join(keyframes_folder, self.RIGHT_DEPTH)

        camera_path = os.path.join(keyframes_folder, self.CAMERA_PARAMETERS)

        left_image = Image.open(left_image_path).convert('RGB')
        right_image = Image.open(right_image_path).convert('RGB')

        left_depth = tifffile.imread(left_depth_path)
        right_depth = tifffile.imread(right_depth_path)

        with open(camera_path) as f:
            camera = yaml.load(f, loader=yaml.Loader)

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
            'focal': torch.tensor(camera['focal_length']),
            'baseline': torch.tensor(camera['baseline'])
        }

    def __len__(self) -> int:
        return len(self.keyframes)