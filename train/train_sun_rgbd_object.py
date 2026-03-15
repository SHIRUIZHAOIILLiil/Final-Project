import torch, random, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import SUNRGBDObjectROIDataset
from models import build_model
from utilities import load_yaml, get_input, ExperimentLogger
from test import evaluate_for_object

KEEP = {"chair","cabinet","table","sofa","bed","lamp","bottle","monitor","sink","tv"}

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

    for batch, y in loader:
        y = y.to(device)
        if mode in ["rgb", "depth"]:
            x = batch.to(device, non_blocking=True)  # (B,3,H,W) or (B,1,H,W)
        elif mode == "rgbd":
            x_rgb, x_d = batch
            x = torch.cat([x_rgb, x_d], dim=1).to(device, non_blocking=True)  # (B,4,H,W)
        else:
            raise ValueError(f"Unknown mode: {mode}")

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

    for batch, y in loader:
        y = y.to(device)

        if mode in ["rgb", "depth"]:
            x = batch.to(device, non_blocking=True)  # (B,3,H,W) or (B,1,H,W)
        elif mode == "rgbd":
            x_rgb, x_d = batch
            x = torch.cat([x_rgb, x_d], dim=1).to(device, non_blocking=True)  # (B,4,H,W)
        else:
            raise ValueError(mode)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs
        correct += (logits.argmax(dim=1) == y).sum().item()

    return loss_sum / n, correct / n

def train(cfg, mode='rgb', epochs : int = 10, batch_size: int = 1, pretrained: bool = True, topk: int = 10, seed_for_train: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = SUNRGBDObjectROIDataset(cfg, split="train", mode=mode, topk=topk)
    ds_val = SUNRGBDObjectROIDataset(cfg, split="val", mode=mode, label2id=ds_train.label2id)
    ds_test = SUNRGBDObjectROIDataset(cfg, split="test", mode=mode, label2id=ds_train.label2id)

    # 10种固定的类别
    ds_train.samples = [s for s in ds_train.samples if s["label"] in KEEP]
    labels = sorted(list(KEEP))
    ds_train.label2id = {lab: i for i, lab in enumerate(labels)}

    ds_val.samples = [s for s in ds_val.samples if s["label"] in KEEP]
    ds_test.samples = [s for s in ds_test.samples if s["label"] in KEEP]
    ds_val.label2id = ds_train.label2id
    ds_test.label2id = ds_train.label2id


    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    num_classes = len(ds_train.label2id)
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

    save_path_model = f"../checkpoints/object_best_{mode}_seed_{cfg["dataset"]["split"]["seed"]}_model_{cfg["dataset"]["model"]["name"]}.pth"
    save_path_outcome = f"../outcomes/object_outcomes_{cfg["dataset"]["model"]["name"]}.csv"
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
            scheduler.step()

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

    _, test_acc = evaluate_for_object(cfg=cfg, mode=mode, modelPath=save_path_model, test=ds_test)
    logger.log(
        mode=mode,
        seed=seed_for_train,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        test_acc=test_acc,
    )

if __name__ == "__main__":
    cfg = load_yaml("./configs/dataset_sun_rgbd_object.yaml")
    cfg["dataset"]["model"]["name"] = "vit"
    training_seeds = [42, 123, 3407]
    for seed in training_seeds:
        set_global_seed(seed)
        train(cfg, mode="rgb", epochs=40, batch_size=32, pretrained=True, topk=10, seed_for_train=seed)
        train(cfg, mode="depth", epochs=40, batch_size=32, pretrained=True, topk=10, seed_for_train=seed)
        train(cfg, mode="rgbd", epochs=40, batch_size=32, pretrained=True, topk=10, seed_for_train=seed)

