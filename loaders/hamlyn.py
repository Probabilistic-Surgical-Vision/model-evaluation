import glob
import os.path

from typing import Dict, Optional
from PIL import Image, ImageFile

from torch import Tensor
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HamlynDataset(Dataset):

    LEFT_PATH = 'image_0'
    RIGHT_PATH = 'image_1'
    EXTENSION = 'png'

    def __init__(self, root: str, split: str,
                 transform: Optional[object] = None,
                 limit: Optional[int] = None) -> None:

        if split not in ('train', 'test'):
            raise ValueError('Split must be either "train" or "test".')

        left_glob = os.path.join(root, split, self.LEFT_PATH,
                                 f'*.{self.EXTENSION}')

        right_glob = os.path.join(root, split, self.RIGHT_PATH,
                                  f'*.{self.EXTENSION}')

        left_images = glob.glob(left_glob)
        right_images = glob.glob(right_glob)

        left_names = set(map(os.path.basename, left_images))
        right_names = set(map(os.path.basename, right_images))

        missing = left_names.symmetric_difference(right_names)

        if len(missing) > 0:
            print(f'Missing {len(missing):,} images from the dataset.')
            left_images = [i for i in left_images if i not in missing]
            right_images = [i for i in right_images if i not in missing]
            print(f'Dataset reduced to {len(left_images):,} images.')

        self.lefts = sorted(left_images[:limit])
        self.rights = sorted(right_images[:limit])

        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        left_path = self.lefts[idx]
        right_path = self.rights[idx]

        left = Image.open(left_path).convert('RGB')
        right = Image.open(right_path).convert('RGB')

        image_pair = {'left': left, 'right': right}

        if self.transform is not None:
            image_pair = self.transform(image_pair)

        return image_pair

    def __len__(self) -> int:
        return len(self.lefts)
