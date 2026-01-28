import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import SUNRGBDSceneDataset
from models import build_resnet18
from utilities import load_yaml

def get_input(rgb, depth, mode: str):
    if mode == "rgb":
        return rgb
    elif mode == "depth":
        return depth
    elif mode == "rgbd":
        return torch.cat([rgb, depth], dim=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def train_one_epoch(model, loader, optimizer, criterion, device, mode: str):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0

    for rgb, depth, y, sid in loader:
        rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)
        x = get_input(rgb, depth, mode)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs
        correct += (logits.argmax(dim=1) == y).sum().item()

    return loss_sum / n, correct / n

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, mode: str):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0

    for rgb, depth, y, sid in loader:
        rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)
        x = get_input(rgb, depth, mode)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs
        correct += (logits.argmax(dim=1) == y).sum().item()

    return loss_sum / n, correct / n

def train(cfg, mode='rgb', epochs : int = 10, batch_size: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = SUNRGBDSceneDataset(cfg, split="train")
    ds_val = SUNRGBDSceneDataset(cfg, split="val")

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    num_classes = len(ds_train.label_to_index)
    in_channels = {"rgb": 3, "depth": 1, "rgbd": 4}[mode]

    model = build_resnet18(num_classes=num_classes, in_channels=in_channels)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    best_val_acc = -1
    patience = 8
    bad_epochs = 0
    min_delta = 1e-3

    save_path = f"./checkpoints/best_{mode}_model.pth"

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, loader_train, optimizer, criterion, device, mode)
        val_loss, val_acc = eval_one_epoch(model, loader_val, criterion, device, mode)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step(val_loss)

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            bad_epochs = 0
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            }, save_path)

        else:
            bad_epochs += 1
        print(bad_epochs)
        if bad_epochs >= patience:
            break


def main():
    cfg = load_yaml("./configs/dataset_sun_rgb_d.yaml")
    train(cfg, mode="depth", epochs=20, batch_size=16)




if __name__ == '__main__':
    main()