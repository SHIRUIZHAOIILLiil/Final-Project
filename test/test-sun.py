import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import SUNRGBDSceneDataset
from models import load_model_test
from utilities import load_yaml, get_input
from pathlib import Path


def evaluate(cfg, mode:str, modelPath:Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_test = SUNRGBDSceneDataset(cfg=cfg, split="test")
    in_channels = {"rgb": 3, "depth": 1, "rgbd": 4}[mode]
    num_classes = len(ds_test.label_to_index)

    batch_size = 16
    model = load_model_test(modelPath,
                       device=device,
                       num_classes=num_classes,
                       in_channels=in_channels)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    test_loss_sum = 0.0
    test_correct = 0
    test_n = 0

    with torch.no_grad():
        for rgb, depth, y, sid in test_loader:
            rgb = rgb.to(device)
            depth = depth.to(device)
            y = y.to(device)

            x = get_input(rgb, depth, mode)
            logits = model(x)
            loss = criterion(logits, y)

            test_loss_sum += loss.item() * y.size(0)
            test_n += y.size(0)

            pred = logits.argmax(dim=1)
            test_correct += (pred == y).sum().item()

    test_loss = test_loss_sum / test_n
    test_acc = test_correct / test_n
    print(f"TEST: loss={test_loss:.4f}, acc={test_acc:.4f}")


if __name__ == "__main__":
    cfg = load_yaml("configs/dataset_sun_rgb_d.yaml")
    evaluate(cfg=cfg, mode="depth", modelPath=Path("../checkpoints/best_depth_model.pth"))








