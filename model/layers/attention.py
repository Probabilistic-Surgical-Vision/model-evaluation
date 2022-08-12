import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EfficientAttention(nn.Module):

    def __init__(self, image_channels: int, key_channels: int,
                 value_channels: int, head_size: int) -> None:

        super().__init__()

        self.image_channels = image_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.head_size = head_size

        self.key_channels_per_head = key_channels // head_size
        self.value_channels_per_head = value_channels // head_size

        self.keys = nn.Conv2d(image_channels, key_channels, 1)
        self.queries = nn.Conv2d(image_channels, key_channels, 1)
        self.values = nn.Conv2d(image_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, image_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.size()
        image_size = height * width

        keys = self.keys(x).reshape(-1, self.key_channels, image_size)
        queries = self.queries(x).reshape(-1, self.key_channels, image_size)
        values = self.values(x).reshape(-1, self.value_channels, image_size)

        attended_values = []

        for i in range(self.head_size):
            key_head_start = i * self.key_channels_per_head
            key_head_end = (i + 1) * self.key_channels_per_head

            value_head_start = i * self.value_channels_per_head
            value_head_end = (i + 1) * self.value_channels_per_head

            key = keys[:, key_head_start:key_head_end]
            query = queries[:, key_head_start:key_head_end]
            value = values[:, value_head_start:value_head_end]

            key = F.softmax(key, dim=2)
            query = F.softmax(query, dim=1)

            context = key @ value.transpose(1, 2)

            attended_value = (context.transpose(1, 2) @ query) \
                .reshape(-1, self.value_channels_per_head, height, width)

            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)

        return reprojected_value + x
