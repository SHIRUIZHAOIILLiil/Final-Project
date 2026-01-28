import torch
from torch.utils.data import DataLoader
from datasets import SUNRGBDSceneDataset
from models import load_model_test
from utilities import load_yaml
import torch.nn as nn

def evaluate():
    cfg = load_yaml("./configs/dataset_sun_rgb_d.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_test = SUNRGBDSceneDataset(cfg=cfg, split="test")
    mode = "rgb"
    in_channels = {"rgb": 3, "depth": 1, "rgbd": 4}[mode]
    num_classes = len(ds_test.label_to_index)

    batch_size = 16
    model = load_model_test('../checkpoints/best_model.pth',
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

            x = rgb  # RGB-only
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
    evaluate()








