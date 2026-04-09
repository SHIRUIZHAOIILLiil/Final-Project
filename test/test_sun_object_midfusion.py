import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import SUNRGBDObjectROIDatasetMidFusion
from models import build_model


def evaluate_for_object_midfusion(cfg, modelPath: str, test: SUNRGBDObjectROIDatasetMidFusion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_test = test
    num_classes = len(ds_test.label2id)
    batch_size = 16

    # Keep in_channels=4 only for compatibility with your existing build_model signature.
    model = build_model(cfg, num_classes=num_classes, in_channels=4, pretrained=False)
    model.to(device)
    model.eval()

    h = cfg["dataset"]["preprocessing"]["image_size"][0]
    w = cfg["dataset"]["preprocessing"]["image_size"][1]
    dummy_rgb = torch.zeros(1, 3, h, w, device=device)
    dummy_d = torch.zeros(1, 1, h, w, device=device)
    with torch.no_grad():
        _ = model((dummy_rgb, dummy_d))

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
            x_rgb, x_d = batch
            x_rgb = x_rgb.to(device, non_blocking=True)
            x_d = x_d.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model((x_rgb, x_d))
            loss = criterion(logits, y)

            test_loss_sum += loss.item() * y.size(0)
            test_n += y.size(0)
            pred = logits.argmax(dim=1)
            test_correct += (pred == y).sum().item()

    test_loss = test_loss_sum / test_n
    test_acc = test_correct / test_n
    return test_loss, test_acc


if __name__ == "__main__":
    pass
