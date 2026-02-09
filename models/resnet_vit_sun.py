import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_vit_b16(num_classes: int, in_channels: int, pretrained: bool = True):
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)


    old = model.conv_proj  # Conv2d(3 -> hidden_dim, kernel_size=16, stride=16)
    if in_channels != old.in_channels:
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )

        with torch.no_grad():
            if pretrained:
                if in_channels == 1:
                    new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                elif in_channels == 4:
                    new_conv.weight[:, :3].copy_(old.weight)
                    new_conv.weight[:, 3:4].copy_(old.weight.mean(dim=1, keepdim=True))
                elif in_channels == 3:
                    new_conv.weight.copy_(old.weight)

                if old.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(old.bias)
            else:
                pass

        model.conv_proj = new_conv
    # output model
    in_dim = model.heads.head.in_features
    model.heads.head = nn.Linear(in_dim, num_classes)

    return model