import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights,
    ViT_B_16_Weights,
    resnet18,
    vit_b_16,
)


def _adapt_conv2d_input(conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    )

    with torch.no_grad():
        if pretrained:
            if in_channels == 1:
                new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            elif in_channels == 3:
                new_conv.weight.copy_(conv.weight)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


class ResNetFeatureEncoder(nn.Module):
    """ResNet18 feature extractor up to layer4."""

    def __init__(self, in_channels: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = resnet18(weights=weights)
        base_model.conv1 = _adapt_conv2d_input(base_model.conv1, in_channels, pretrained=weights is not None)

        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ViTFeatureEncoder(nn.Module):
    """ViT-B/16 feature extractor that returns patch tokens as a feature map."""

    def __init__(self, in_channels: int, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        self.backbone.conv_proj = _adapt_conv2d_input(
            self.backbone.conv_proj,
            in_channels,
            pretrained=weights is not None,
        )
        self.patch_size = self.backbone.patch_size
        self.out_channels = self.backbone.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"ViT branch expects input sizes divisible by patch size {self.patch_size}, "
                f"got {(height, width)}"
            )

        tokens = self.backbone._process_input(x)
        cls_token = self.backbone.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = self.backbone.encoder(tokens)
        tokens = tokens[:, 1:, :]

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        return tokens.transpose(1, 2).reshape(batch_size, self.out_channels, grid_h, grid_w)


def _build_branch_encoder(backbone: str, in_channels: int, pretrained: bool) -> nn.Module:
    if backbone == "resnet18":
        return ResNetFeatureEncoder(in_channels=in_channels, pretrained=pretrained)
    if backbone == "vit":
        return ViTFeatureEncoder(in_channels=in_channels, pretrained=pretrained)
    raise ValueError(f"Unsupported mid-fusion backbone: {backbone}")


class DualBackboneMidFusion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        rgb_backbone: str = "resnet18",
        depth_backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.1,
        layers: int = 2,
        heads: int = 8,
        dim_feedforward: int = 2048,
        fusion: str = "concat",
        fusion_dim: int = 512,
        fusion_grid_size: int = 7,
        head_type: str = "transformer",
    ):
        super().__init__()
        if fusion not in ("concat", "add", "gated"):
            raise ValueError(f"Unsupported fusion='{fusion}'. Use 'concat', 'add', or 'gated'.")
        if head_type not in ("transformer", "mlp"):
            raise ValueError(f"Unsupported head_type='{head_type}'. Use 'transformer' or 'mlp'.")

        self.embed_dim = fusion_dim
        self.fusion = fusion
        self.fusion_grid_size = fusion_grid_size
        self.head_type = head_type
        self.rgb_backbone_name = rgb_backbone
        self.depth_backbone_name = depth_backbone

        self.rgb_encoder = _build_branch_encoder(rgb_backbone, in_channels=3, pretrained=pretrained)
        self.depth_encoder = _build_branch_encoder(depth_backbone, in_channels=1, pretrained=pretrained)

        self.rgb_proj = self._make_branch_proj(self.rgb_encoder.out_channels)
        self.depth_proj = self._make_branch_proj(self.depth_encoder.out_channels)

        if fusion == "concat":
            self.fuse_proj = nn.Sequential(
                nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.embed_dim),
                nn.ReLU(inplace=True),
            )
        elif fusion == "gated":
            self.gate = nn.Sequential(
                nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = None
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.head_type == "transformer":
            enc = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
            self.norm = nn.LayerNorm(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.norm = nn.LayerNorm(self.embed_dim)
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, num_classes),
            )

    def _make_branch_proj(self, in_channels: int) -> nn.Module:
        if in_channels == self.embed_dim:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv2d(in_channels, self.embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
        )

    def _pool_and_project(self, features: torch.Tensor, projector: nn.Module) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(features, output_size=(self.fusion_grid_size, self.fusion_grid_size))
        return projector(pooled)

    def _build_pos_embed_if_needed(self, token_len: int, device: torch.device):
        if self.pos_embed is None or self.pos_embed.shape[1] != token_len:
            pe = nn.Parameter(torch.zeros(1, token_len, self.embed_dim, device=device))
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = pe

    @staticmethod
    def _unpack_inputs(inputs):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("Dual-input mid-fusion model expects inputs as (x_rgb, x_depth).")
        x_rgb, x_depth = inputs
        if x_rgb.dim() != 4 or x_rgb.size(1) != 3:
            raise ValueError(f"Expected x_rgb shape (B, 3, H, W), got {tuple(x_rgb.shape)}")
        if x_depth.dim() != 4 or x_depth.size(1) != 1:
            raise ValueError(f"Expected x_depth shape (B, 1, H, W), got {tuple(x_depth.shape)}")
        return x_rgb, x_depth

    def _fuse_features(self, rgb_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        if self.fusion == "concat":
            return self.fuse_proj(torch.cat([rgb_features, depth_features], dim=1))
        if self.fusion == "add":
            return rgb_features + depth_features
        gate = self.gate(torch.cat([rgb_features, depth_features], dim=1))
        return rgb_features + gate * depth_features

    def _forward_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        if self.head_type == "mlp":
            pooled = self.pool(fused).flatten(1)
            pooled = self.norm(pooled)
            return self.head(pooled)

        batch_size = fused.size(0)
        tokens = fused.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        self._build_pos_embed_if_needed(tokens.size(1), tokens.device)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])

    def forward(self, inputs) -> torch.Tensor:
        x_rgb, x_depth = self._unpack_inputs(inputs)
        rgb_features = self._pool_and_project(self.rgb_encoder(x_rgb), self.rgb_proj)
        depth_features = self._pool_and_project(self.depth_encoder(x_depth), self.depth_proj)
        fused = self._fuse_features(rgb_features, depth_features)
        return self._forward_from_fused(fused)


class ResNet18ViTMidFusion(DualBackboneMidFusion):
    """Compatibility wrapper for concat/add mid-fusion variants."""

    def __init__(self, num_classes: int, fusion: str = "concat", **kwargs):
        super().__init__(num_classes=num_classes, fusion=fusion, **kwargs)


class ResNet18ViTMidFusionGated(DualBackboneMidFusion):
    """Compatibility wrapper for gated mid-fusion variants."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes=num_classes, fusion="gated", **kwargs)


def build_resnet_vit_midfusion(
    num_classes: int,
    pretrained: bool = True,
    layers: int = 2,
    heads: int = 8,
    dropout: float = 0.1,
    dim_feedforward: int = 2048,
    fusion: str = "concat",
    rgb_backbone: str = "resnet18",
    depth_backbone: str = "resnet18",
    fusion_dim: int = 512,
    fusion_grid_size: int = 7,
    head_type: str = "transformer",
):
    return ResNet18ViTMidFusion(
        num_classes=num_classes,
        pretrained=pretrained,
        layers=layers,
        heads=heads,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
        fusion=fusion,
        rgb_backbone=rgb_backbone,
        depth_backbone=depth_backbone,
        fusion_dim=fusion_dim,
        fusion_grid_size=fusion_grid_size,
        head_type=head_type,
    )


def build_resnet_vit_midfusion_gated(
    num_classes: int,
    pretrained: bool = True,
    layers: int = 2,
    heads: int = 8,
    dropout: float = 0.1,
    dim_feedforward: int = 2048,
    rgb_backbone: str = "resnet18",
    depth_backbone: str = "resnet18",
    fusion_dim: int = 512,
    fusion_grid_size: int = 7,
    head_type: str = "transformer",
):
    return ResNet18ViTMidFusionGated(
        num_classes=num_classes,
        pretrained=pretrained,
        layers=layers,
        heads=heads,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
        rgb_backbone=rgb_backbone,
        depth_backbone=depth_backbone,
        fusion_dim=fusion_dim,
        fusion_grid_size=fusion_grid_size,
        head_type=head_type,
    )
