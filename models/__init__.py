from .resnet_sun import build_resnet18
from .resnet_vit_sun import build_vit_b16


def build_model(cfg, num_classes: int, in_channels: int, pretrained: bool = True):
    name = cfg["dataset"]["model"]["name"]

    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)

    if name == "vit":
        return build_vit_b16(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)


    raise ValueError(f"Unknown model name: {name}")