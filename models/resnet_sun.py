import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes: int, in_channels: int):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    old = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model_test(checkpoint_path, device='cuda', num_classes: int=0, in_channels: int=0):
    model = build_resnet18(num_classes=num_classes, in_channels=in_channels)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
