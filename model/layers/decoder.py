from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

KernelSize = Union[int, Tuple[int, int]]


class ConvLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 padding: bool = True, reflection: bool = True,
                 sigmoid: bool = False, kernel_size: KernelSize = 3) -> None:

        super().__init__()

        if padding:
            self.padding = nn.ReflectionPad2d(1) \
                if reflection else nn.ZeroPad2d(1)
        else:
            self.padding = None

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.Sigmoid() if sigmoid else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.padding is not None:
            x = self.padding(x)

        return self.layers(x)


class ConvELUBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 padding: bool = True, kernel_size: KernelSize = 3,
                 batch_norm: bool = False) -> None:

        super().__init__()

        self.layers = nn.Sequential(
            ConvLayer(in_channels, out_channels, padding=padding,
                      kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ELU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class SELayer(nn.Module):

    def __init__(self, channels: int, reduction: int = 16,
                 fc: bool = True) -> None:

        super().__init__()

        self.fc = fc
        self.channels_reduced = channels // reduction

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excite = nn.Sequential(
            nn.Linear(channels, self.channels_reduced, bias=False)
            if fc else nn.Conv2d(channels, self.channels_reduced,
                                 kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels_reduced, channels, bias=False)
            if fc else nn.Conv2d(self.channels_reduced, channels,
                                 kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x_squeezed = self.squeeze(x)

        if self.fc:
            b, c, _, _ = x.size()
            x_squeezed = x_squeezed.view(b, c)
            x_excited = self.excite(x_squeezed) \
                .view(b, c, 1, 1) \
                .expand_as(x)
        else:
            x_excited = self.excite(x_squeezed)

        return x * x_excited


class DecoderStage(nn.Module):

    DecoderOut = Tuple[Tensor, Tensor, Optional[Tensor]]

    def __init__(self, in_channels: int, feature_in_channels: int,
                 skip_in_channels: int, upsample_channels: int,
                 out_channels: int, skip_out_channels: int,
                 disp_channels: int = 2, batch_norm: bool = True,
                 fc: bool = True, scale: int = 2, concat_disp: bool = True,
                 calculate_disp: bool = True) -> None:

        super().__init__()

        self.scale = scale
        self.calculate_disp = calculate_disp
        self.concat_disp = concat_disp

        self.upsample = nn.Sequential(
            ConvELUBlock(in_channels, upsample_channels * int(scale ** 2),
                         batch_norm=batch_norm),
            nn.PixelShuffle(upscale_factor=self.scale)
        )

        self.squeeze_excite = nn.Sequential(
            ConvELUBlock(feature_in_channels + skip_in_channels,
                         skip_out_channels, kernel_size=1,
                         batch_norm=True, padding=False),
            SELayer(channels=skip_out_channels, fc=fc)
        )

        iconv_in_channels = upsample_channels + skip_out_channels
        iconv_in_channels += disp_channels if concat_disp else 0

        self.iconv = ConvELUBlock(iconv_in_channels, out_channels,
                                  batch_norm=batch_norm)

        self.disp = ConvLayer(out_channels, disp_channels, sigmoid=True) \
            if self.calculate_disp else None

    def forward(self, x: Tensor, feature_map: Tensor, skip: Tensor,
                disparity: Optional[Tensor] = None,
                scale: Optional[float] = 1.0) -> DecoderOut:

        skip = F.interpolate(skip, scale_factor=self.scale,
                             align_corners=True, mode='bilinear')

        skip = self.squeeze_excite(torch.cat((feature_map, skip), 1))

        x_upsampled = self.upsample(x)
        x_concat = torch.cat((x_upsampled, skip), 1)

        if self.concat_disp:
            disparity = F.interpolate(disparity, scale_factor=self.scale,
                                      align_corners=True, mode='bilinear')

            x_concat = torch.cat((x_concat, disparity), 1)

        out = self.iconv(x_concat)

        disparity = scale * self.disp(out) \
            if self.calculate_disp else None

        return out, skip, disparity
