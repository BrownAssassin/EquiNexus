# Copyright (C) 2026 EquiNexus contributors
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


DINO_V2_REPOSITORY = (
    "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)


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


class DinoV2FilamentUNet(nn.Module):
    """Frozen self-supervised DINOv2 features with a trainable dense decoder."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 16,
        *,
        pretrained_backbone: bool = False,
        hub_repository: str = DINO_V2_REPOSITORY,
    ) -> None:
        super().__init__()
        if in_channels < 1 or base_channels < 4:
            raise ValueError("in_channels must be positive and base_channels at least 4")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.hub_repository = hub_repository
        self.backbone = torch.hub.load(
            hub_repository,
            "dinov2_vits14",
            pretrained=pretrained_backbone,
            trust_repo=True,
        )
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.encoder1 = ConvBlock(in_channels, widths[0])
        self.encoder2 = ConvBlock(widths[0], widths[1])
        self.encoder3 = ConvBlock(widths[1], widths[2])
        self.encoder4 = ConvBlock(widths[2], widths[3])
        self.pool = nn.MaxPool2d(2)

        feature_width = base_channels * 4
        self.feature_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(384, feature_width, kernel_size=1, bias=False),
                nn.GroupNorm(_group_count(feature_width), feature_width),
                nn.SiLU(inplace=True),
            )
            for _ in range(4)
        )
        self.fusion = ConvBlock(widths[3] + feature_width * 4, widths[3])
        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(widths[2] * 2, widths[2])
        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(widths[1] * 2, widths[1])
        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(widths[0] * 2, widths[0])
        self.head = nn.Conv2d(widths[0], 2, kernel_size=1)

    def train(self, mode: bool = True) -> DinoV2FilamentUNet:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, received {image.shape[1]}"
            )
        height, width = image.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError("DINOv2 input height and width must be divisible by 14")

        raw = image[:, 0:1].float().clamp(-3.0, 3.0)
        gray = (raw + 3.0) / 6.0
        rgb = gray.expand(-1, 3, -1, -1)
        mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        backbone_input = (rgb - mean) / std
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                backbone_input,
                n=(2, 5, 8, 11),
                reshape=True,
                norm=True,
            )

        skip1 = self.encoder1(image)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        deep = self.encoder4(self.pool(skip3))
        projected = [layer(feature) for layer, feature in zip(
            self.feature_projections, features, strict=True
        )]
        projected = [
            F.interpolate(feature, size=deep.shape[-2:], mode="bilinear", align_corners=False)
            for feature in projected
        ]
        value = self.fusion(torch.cat((deep, *projected), dim=1))
        value = self.up3(value)
        value = F.interpolate(value, size=skip3.shape[-2:], mode="bilinear", align_corners=False)
        value = self.decoder3(torch.cat((skip3, value), dim=1))
        value = self.up2(value)
        value = F.interpolate(value, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        value = self.decoder2(torch.cat((skip2, value), dim=1))
        value = self.up1(value)
        value = F.interpolate(value, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.decoder1(torch.cat((skip1, value), dim=1))
        return self.head(value)


def build_filament_model(
    *,
    model_type: str = "unet",
    in_channels: int = 1,
    base_channels: int = 16,
    pretrained_backbone: bool = False,
    hub_repository: str = DINO_V2_REPOSITORY,
) -> nn.Module:
    if model_type == "unet":
        return FilamentUNet(in_channels=in_channels, base_channels=base_channels)
    if model_type == "smp_unet_resnet34":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-segmentation.txt to use the SMP ResNet34 U-Net."
            ) from exc
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained_backbone else None,
            in_channels=in_channels,
            classes=2,
        )
        model.in_channels = in_channels
        return model
    if model_type == "dinov2_vits14":
        return DinoV2FilamentUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            pretrained_backbone=pretrained_backbone,
            hub_repository=hub_repository,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def model_architecture(model_type: str) -> str:
    architectures = {
        "unet": "from_scratch_five_level_unet_foreground_boundary",
        "smp_unet_resnet34": "smp_unet_resnet34_foreground_boundary",
        "dinov2_vits14": "dinov2_vits14_frozen_backbone_cnn_decoder_foreground_boundary",
    }
    try:
        return architectures[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported model type: {model_type}") from exc
