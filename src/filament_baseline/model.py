from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class FilamentUNet(nn.Module):
    """Small from-scratch U-Net with foreground and instance-boundary heads."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        widths = [base_channels * (2**index) for index in range(5)]
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths[:-1]:
            self.encoders.append(ConvBlock(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[-2], widths[-1])
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = widths[-1]
        for width in reversed(widths[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(current, width, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width))
            current = width
        self.head = nn.Conv2d(widths[0], 2, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips = []
        value = image
        for encoder in self.encoders:
            value = encoder(value)
            skips.append(value)
            value = self.pool(value)
        value = self.bottleneck(value)
        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, reversed(skips), strict=True
        ):
            value = upconv(value)
            if value.shape[-2:] != skip.shape[-2:]:
                value = nn.functional.interpolate(
                    value, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            value = decoder(torch.cat((skip, value), dim=1))
        return self.head(value)
