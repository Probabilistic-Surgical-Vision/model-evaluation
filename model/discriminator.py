from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .layers.encoder import EncoderStage

KernelSize = Union[int, Tuple[int, int]]
ImagePyramid = List[Tensor]


class RandomDiscriminator(nn.Module):
    def __init__(self, layers: List[dict], final_conv: dict,
                 linear_in_features: int, load_graph: Optional[str] = None,
                 nodes: int = 5, seed: int = 42) -> None:

        super().__init__()

        self.layers = nn.ModuleList()

        for i, layer_config in enumerate(layers):
            self.layers.append(EncoderStage(**layer_config, stage=(i+1),
                                            nodes=nodes, seed=seed,
                                            load_graph=load_graph))

        self.conv = EncoderStage(**final_conv,
                                 stage=(len(self.layers)+1),
                                 nodes=nodes, seed=seed,
                                 load_graph=load_graph)

        self.linear = nn.Linear(linear_in_features, 1)

    def features(self, pyramid: ImagePyramid) -> ImagePyramid:
        features = []

        pyramid_iterator = zip(pyramid, self.layers)
        for i, (images, layer) in enumerate(pyramid_iterator):
            if i == 0:
                out = layer(images)
            else:
                feature_concat = torch.cat((out, images), dim=1)
                out = layer(feature_concat)

            features.append(out)

        return features

    def forward(self, pyramid: ImagePyramid) -> Tensor:
        feature = self.features(pyramid)[-1]

        out = self.conv(feature)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return torch.sigmoid(out)
