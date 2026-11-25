"""Super-resolution neural network architectures for aerial imagery.

All models accept an LR tensor of shape (B, C, H, W) in [0, 1] and output
an HR tensor of shape (B, C, H * scale_factor, W * scale_factor).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network (Shi et al., 2016)."""

    def __init__(self, upscale_factor: int = 4, channels: int = 3) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, channels * (upscale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x


class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network (Dong et al., 2014).

    Upsamples LR input via bicubic interpolation, then applies 9x9 -> 5x5 -> 5x5
    convolutions with ReLU activations.
    """

    def __init__(self, upscale_factor: int = 4, channels: int = 3) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode="bicubic",
            align_corners=False,
        )
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.conv3(x)
        return x


class FSRCNN(nn.Module):
    """Fast Super-Resolution Convolutional Neural Network (Dong et al., 2016)."""

    def __init__(
        self,
        upscale_factor: int = 4,
        channels: int = 3,
        d: int = 56,
        s: int = 12,
        m: int = 4,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

        self.feature_extract = nn.Sequential(
            nn.Conv2d(channels, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        )
        mapping_layers: list[nn.Module] = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU(s))
        self.mapping = nn.Sequential(*mapping_layers)

        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
        )
        self.deconv = nn.ConvTranspose2d(
            d,
            channels,
            kernel_size=9,
            stride=upscale_factor,
            padding=4,
            output_padding=upscale_factor - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extract(x)
        x = self.shrink(x)
        x = self.mapping(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResidualSRNet(nn.Module):
    """Lightweight Residual SR Network with PixelShuffle and global bicubic skip."""

    def __init__(
        self,
        upscale_factor: int = 4,
        channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.head = nn.Conv2d(channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[_ResBlock(num_features) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, channels * (upscale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode="bicubic",
            align_corners=False,
        )
        feat = self.head(x)
        res = self.body(feat)
        out = self.tail(feat + res)
        return base + out
