import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights, resnet18, ResNet18_Weights


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


class ResNet18ViT(nn.Module):
    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            pretrained: bool = True,
            dropout: float = 0.1,
            layers: int = 2,
            heads: int = 8,
            dim_feedforward: int = 2048,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = resnet18(weights=weights)

        old = base_model.conv1
        if in_channels != old.in_channels:
            new_conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=(old.bias is not None),
            )
            with torch.no_grad():
                if weights is not None:
                    if in_channels == 1:
                        new_conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                    elif in_channels == 4:
                        new_conv1.weight[:, :3].copy_(old.weight)
                        new_conv1.weight[:, 3:4].copy_(old.weight.mean(dim=1, keepdim=True))
                    elif in_channels == 3:
                        new_conv1.weight.copy_(old.weight)
                else:
                    nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
                if old.bias is not None and new_conv1.bias is not None:
                    new_conv1.bias.copy_(old.bias)
            base_model.conv1 = new_conv1
        # Extract the "first four layers of the basic structure" of ResNet and save them as a separate module.
        self.stem = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        # Get the backbone of Resnet18
        self.layer1, self.layer2, self.layer3, self.layer4 = base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4

        embed_dim = 512  # ResNet18 layer4 channels
        self.embed_dim = embed_dim

        # CLS + pos embed (pos created lazily)输出是一串 token，如果要做整图分类，得把第四层的输出换成token。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None # 根据 token_len 创建 pos_embed
        # Initialize CLS token, use a normal distribution with small variance
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Core of VIT
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)# 把上面的 encoder layer 复制 layers 次，比如 2 层。
        # 1 层：每个 token 能“看一眼”所有 token
        # 2 层：看完后再“思考”一次（更强的全局建模）
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _build_pos_embed_if_needed(self, token_len: int, device):
        if self.pos_embed is None or self.pos_embed.shape[1] != token_len:
            pe = nn.Parameter(torch.zeros(1, token_len, self.embed_dim, device=device))
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, H', W')

        x = x.flatten(2).transpose(1, 2)  # (B, HW, 512)

        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, 512)
        # Add a classification token,专门用来“收集整张图信息”的 token
        x = torch.cat([cls, x], dim=1)  # (B, 1+HW, 512)

        # 告诉 Transformer 每个 token 在哪
        self._build_pos_embed_if_needed(x.size(1), x.device)
        x = x + self.pos_embed

        x = self.transformer(x)
        x = self.norm(x)
        # 表示整张图的全局语义
        logits = self.head(x[:, 0])
        return logits


def build_resnet_vit(
        num_classes: int,
        in_channels: int,
        pretrained: bool = True,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
):
    return ResNet18ViT(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        layers=layers,
        heads=heads,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
    )
