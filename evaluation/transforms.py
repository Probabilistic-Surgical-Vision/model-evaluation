from typing import Dict, Tuple

from torch import Tensor
from torchvision import transforms

ImageDict = Dict[str, Tensor]
BoundsTuple = Tuple[float, float]
ImageSizeTuple = Tuple[int, int]


class ResizeImage:

    def __init__(self, size: ImageSizeTuple = (256, 512)) -> None:
        self.transform = transforms.Resize(size)

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        left = self.transform(image_pair["left"])
        right = self.transform(image_pair["right"])

        return {"left": left, "right": right}


class ToTensor:

    def __init__(self) -> None:
        self.transform = transforms.ToTensor()

    def __call__(self, image_pair: ImageDict) -> ImageDict:
        left = self.transform(image_pair["left"])
        right = self.transform(image_pair["right"])

        return {"left": left, "right": right}
