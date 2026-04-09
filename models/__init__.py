from .resnet_sun import build_resnet18
from .resnet_vit_sun import build_vit_b16, build_resnet_vit
from .resnet_vit_sun_midfusion import build_resnet_vit_midfusion, build_resnet_vit_midfusion_gated


def build_model(cfg, num_classes: int, in_channels: int, pretrained: bool = True):
    name = cfg["dataset"]["model"]["name"]
    model_cfg = cfg["dataset"].get("model", {})

    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)

    if name == "vit":
        return build_vit_b16(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)

    if name == "resnet18ViT":
        return build_resnet_vit(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)

    if name == "resnet18vit_midfusion":
        return build_resnet_vit_midfusion(
            num_classes=num_classes,
            pretrained=pretrained,
            rgb_backbone=model_cfg.get("rgb_backbone", "resnet18"),
            depth_backbone=model_cfg.get("depth_backbone", "resnet18"),
            dropout=model_cfg.get("dropout", 0.1),
            fusion_dim=model_cfg.get("fusion_dim", 512),
            fusion_grid_size=model_cfg.get("fusion_grid_size", 7),
            head_type=model_cfg.get("head_type", "transformer"),
        )

    if name == "resnet18vit_midfusionGated":
        return build_resnet_vit_midfusion_gated(
            num_classes=num_classes,
            pretrained=pretrained,
            rgb_backbone=model_cfg.get("rgb_backbone", "resnet18"),
            depth_backbone=model_cfg.get("depth_backbone", "resnet18"),
            dropout=model_cfg.get("dropout", 0.1),
            fusion_dim=model_cfg.get("fusion_dim", 512),
            fusion_grid_size=model_cfg.get("fusion_grid_size", 7),
            head_type=model_cfg.get("head_type", "transformer"),
        )


    raise ValueError(f"Unknown model name: {name}")
