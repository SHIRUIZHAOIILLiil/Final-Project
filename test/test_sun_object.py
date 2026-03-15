import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import SUNRGBDObjectROIDataset
from models import build_model
from utilities import load_yaml, get_input


def evaluate_for_object(cfg, mode:str, modelPath:str, test: SUNRGBDObjectROIDataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_test = test
    in_channels = {"rgb": 3, "depth": 1, "rgbd": 4}[mode]
    num_classes = len(ds_test.label2id)

    batch_size = 16

    model = build_model(cfg, num_classes=num_classes, in_channels=in_channels, pretrained=False)

    model.to(device)
    model.eval()
    dummy = torch.zeros(1, in_channels, cfg["dataset"]["preprocessing"]["image_size"][0], cfg["dataset"]["preprocessing"]["image_size"][1], device=device)
    with torch.no_grad():
        _ = model(dummy)

    ckpt = torch.load(modelPath, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0001)

    test_loss_sum = 0.0
    test_correct = 0
    test_n = 0

    with torch.no_grad():
        for batch, y in test_loader:
            y = y.to(device)

            if mode in ["rgb", "depth"]:
                x = batch.to(device, non_blocking=True)
            elif mode == "rgbd":
                x_rgb, x_d = batch
                x = torch.cat([x_rgb, x_d], dim=1).to(device, non_blocking=True)
            else:
                raise ValueError(f"Unknown mode: {mode}")

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
    pass









