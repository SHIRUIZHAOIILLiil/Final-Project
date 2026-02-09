import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes: int, in_channels: int, pretrained: bool = True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    old = model.conv1
    if in_channels != old.in_channels:
        new_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )

        with torch.no_grad():
            if in_channels == 1:
                # 把 RGB 三通道权重平均成单通道
                new_conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            elif in_channels == 4:
                new_conv1.weight[:, :3].copy_(old.weight)
                new_conv1.weight[:, 3:4].copy_(old.weight.mean(dim=1, keepdim=True))
            elif in_channels == 3:
                new_conv1.weight.copy_(old.weight)

            if old.bias is not None and new_conv1.bias is not None:
                new_conv1.bias.copy_(old.bias)

        model.conv1 = new_conv1

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
