import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import SUNRGBDSceneDataset
from models import build_model
from utilities import load_yaml, get_input


def evaluate(cfg, mode:str, modelPath:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_test = SUNRGBDSceneDataset(cfg=cfg, split="test")
    in_channels = {"rgb": 3, "depth": 1, "rgbd": 4}[mode]
    num_classes = len(ds_test.label_to_index)

    batch_size = 16

    model = build_model(cfg, num_classes=num_classes, in_channels=in_channels, pretrained=False)
    ckpt = torch.load(modelPath, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0001)

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
    # print(f"TEST: loss={test_loss:.4f}, acc={test_acc:.4f}")
    return test_loss, test_acc


if __name__ == "__main__":
    cfg = load_yaml("configs/dataset_sun_rgb_d.yaml")
    # evaluate(cfg=cfg, mode="rgb", modelPath=Path("../checkpoints/best_rgb_model.pth"))
    # evaluate(cfg=cfg, mode="depth", modelPath=Path("../checkpoints/best_depth_model.pth"))
    evaluate(cfg=cfg, mode="rgbd", modelPath="checkpoints/best_rgbd_model.pth")









