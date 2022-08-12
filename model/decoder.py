from typing import List, Tuple, Union

import torch.nn as nn
from torch import Tensor

from .layers.decoder import DecoderStage

DecoderOut = Union[Tuple[Tensor, ...], Tensor]


class DepthDecoder(nn.Module):

    def __init__(self, layers: List[dict]) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for layer_config in layers:
            self.layers.append(DecoderStage(**layer_config))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, left_image: Tensor, *feature_maps: Tensor,
                scale: float = 1.0) -> DecoderOut:

        f1, f2, f3, f4, x4 = feature_maps

        out5, skip5, _ = self.layers[0](x4, f4, x4, scale=scale)

        out4, skip4, disp4 = self.layers[1](out5, f3, skip5, scale=scale)
        out3, skip3, disp3 = self.layers[2](out4, f2, skip4, disp4, scale)
        out2, skip2, disp2 = self.layers[3](out3, f1, skip3, disp3, scale)

        _, _, disp1 = self.layers[4](out2, left_image, skip2, disp2, scale)

        if self.training:
            return disp1, disp2, disp3, disp4

        return disp1
