from typing import List, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from .layers.encoder import EncoderStage


class RandomEncoder(nn.Module):
    def __init__(self, layers: List[dict], load_graph: Optional[str] = None,
                 nodes: int = 5, seed: int = 42) -> None:

        super().__init__()

        self.layers = nn.ModuleList()

        for i, layer_config in enumerate(layers):
            self.layers.append(EncoderStage(**layer_config, stage=(i+1),
                                            nodes=nodes, seed=seed,
                                            load_graph=load_graph))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        encodings = []

        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(x)
            else:
                out = layer(out)

            encodings.append(out)

        return tuple(encodings)
