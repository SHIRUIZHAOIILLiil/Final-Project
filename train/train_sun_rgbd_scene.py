import torch, random, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import SUNRGBDSceneDataset
from models import build_model
from utilities import load_yaml, get_input, ExperimentLogger
from test import evaluate

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_seeds(base_cfg: dict):
    base_seed = base_cfg["dataset"]["split"]["seed"]
    seeds = [424, 9, base_cfg["dataset"]["split"]["seed"]]
    seeds = [base_seed] + [s for s in seeds if s != base_seed]
    return seeds

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

def train(cfg, mode='rgb', epochs : int = 10, batch_size: int = 1, pretrained: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = SUNRGBDSceneDataset(cfg, split="train")
    ds_val = SUNRGBDSceneDataset(cfg, split="val")

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    num_classes = len(ds_train.label_to_index)
    in_channels = {"rgb": 3, "depth": 1, "rgbd": 4}[mode]

    model = build_model(cfg, num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)# build_resnet18(num_classes=num_classes, in_channels=in_channels)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    best_val_acc = -1
    patience = 8
    bad_epochs = 0
    min_delta = 1e-3
    best_epoch = -1

    save_path_model = f"../checkpoints/best_{mode}_seed_{cfg["dataset"]["split"]["seed"]}_model_{cfg["dataset"]["model"]["name"]}.pth"
    save_path_outcome = f"../outcomes/outcomes_{cfg["dataset"]["model"]["name"]}.csv"
    logger = ExperimentLogger(save_path_outcome)

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
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            }, save_path_model)

        else:
            bad_epochs += 1
        print(bad_epochs)
        if bad_epochs >= patience:
            break
    _, test_acc = evaluate(cfg=cfg, mode=mode, modelPath=save_path_model)
    logger.log(
        mode=mode,
        seed=cfg["dataset"]["split"]["seed"],
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        test_acc=test_acc,
    )

def main(model:str = "resnet18"):
    base_cfg = load_yaml("./configs/dataset_sun_rgb_d.yaml")
    seeds = set_seeds(base_cfg)

    for seed in seeds:
        cfg = load_yaml("./configs/dataset_sun_rgb_d.yaml")

        cfg["dataset"]["split"]["seed"] = seed
        cfg["dataset"]["model"]["name"] = model

        set_global_seed(seed)

        train(cfg, mode="rgb", epochs=40, batch_size=16)
        train(cfg, mode="depth", epochs=40, batch_size=16)
        train(cfg, mode="rgbd", epochs=40, batch_size=16)




if __name__ == '__main__':
    # main()
    # cfg = load_yaml("./configs/dataset_sun_rgb_d.yaml")
    # cfg["dataset"]["model"]["name"] = "vit"
    # train(cfg, mode="depth", epochs=5, batch_size=16)
    main("vit")