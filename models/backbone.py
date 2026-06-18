"""
models/backbone.py
------------------
CNN feature extractor for single-channel silhouette frames.

Design decisions:
- We use standard ResNet variants (18/34/50) pre-trained on ImageNet.
- The first convolutional layer is replaced with a 1-channel version because
  CASIA-B silhouettes are grayscale.  Weights are preserved by summing the
  three input channels of the original conv1 weights.
- The final classification head (fc layer) is removed; we use the global
  average-pooled feature vector as the per-frame embedding.
- Output dim: 512 for ResNet-18/34, 2048 for ResNet-50.
  (The transformer 'hidden_dim' projection handles dimensional reduction.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


_BACKBONE_OUTPUT_DIM = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


class CNNBackbone(nn.Module):
    """
    ResNet feature extractor adapted for 1-channel silhouette input.

    Args:
        backbone_name: 'resnet18' | 'resnet34' | 'resnet50'
        pretrained:    Load ImageNet weights (True by default).
                       The 1-channel adaptation sums the 3 input-channel
                       weights so no information from pre-training is lost.

    Forward:
        Input:  (B, 1, H, W)
        Output: (B, D)  where D = 512 or 2048
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()

        if backbone_name not in _BACKBONE_OUTPUT_DIM:
            raise ValueError(
                f"backbone_name must be one of {list(_BACKBONE_OUTPUT_DIM.keys())},"
                f" got '{backbone_name}'"
            )

        self.name       = backbone_name
        self.output_dim = _BACKBONE_OUTPUT_DIM[backbone_name]

        # ---- Load base model -----------------------------------------------
        weights_enum = {
            "resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2,
        }
        if pretrained:
            base = getattr(tvm, backbone_name)(weights=weights_enum[backbone_name])
        else:
            base = getattr(tvm, backbone_name)(weights=None)

        # ---- Adapt conv1 for single-channel input --------------------------
        # Original conv1: (64, 3, 7, 7)
        # New conv1:      (64, 1, 7, 7)  — sum over input channel dimension
        orig_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        with torch.no_grad():
            # Sum the three RGB filter banks into one channel
            # This is a principled initialisation: the sum preserves the
            # linear projection learned on ImageNet luminance-like signals.
            new_conv.weight.copy_(orig_conv.weight.sum(dim=1, keepdim=True))
        base.conv1 = new_conv

        # ---- Remove the classification FC layer ----------------------------
        # Replace with Identity; we'll use the global avg-pool features.
        self.feature_extractor = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W)

        Returns:
            features: (B, D)
        """
        feats = self.feature_extractor(x)   # (B, D, 1, 1)
        feats = feats.flatten(1)            # (B, D)
        return feats
