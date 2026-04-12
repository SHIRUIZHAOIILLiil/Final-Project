import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import SUNRGBDObjectROIDatasetMidFusionLowLight
from models import build_model
from utilities import load_yaml, ExperimentLogger, apply_experiment_preset
from test.test_sun_object_midfusion import evaluate_for_object_midfusion

KEEP = {"chair", "cabinet", "table", "sofa", "bed", "lamp", "bottle", "monitor", "sink", "tv"}


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_one_epoch_midfusion(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0

    for batch, y in loader:
        x_rgb, x_d = batch
        x_rgb = x_rgb.to(device, non_blocking=True)
        x_d = x_d.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model((x_rgb, x_d))
        loss = criterion(logits, y)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs
        correct += (logits.argmax(dim=1) == y).sum().item()

    return loss_sum / n, correct / n


def train_one_epoch_midfusion(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0

    for batch, y in loader:
        x_rgb, x_d = batch
        x_rgb = x_rgb.to(device, non_blocking=True)
        x_d = x_d.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model((x_rgb, x_d))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs
        correct += (logits.argmax(dim=1) == y).sum().item()

    return loss_sum / n, correct / n


def _filter_keep_classes(ds_train, ds_val, ds_test):
    ds_train.samples = [s for s in ds_train.samples if s["label"] in KEEP]
    labels = sorted(list(KEEP))
    ds_train.label2id = {lab: i for i, lab in enumerate(labels)}

    ds_val.samples = [s for s in ds_val.samples if s["label"] in KEEP]
    ds_test.samples = [s for s in ds_test.samples if s["label"] in KEEP]
    ds_val.label2id = ds_train.label2id
    ds_test.label2id = ds_train.label2id


def _midfusion_experiment_tag(cfg) -> str:
    model_cfg = cfg["dataset"]["model"]
    rgb_backbone = model_cfg.get("rgb_backbone", "resnet18")
    depth_backbone = model_cfg.get("depth_backbone", "resnet18")
    head_type = model_cfg.get("head_type", "transformer")
    return f"{rgb_backbone}_{depth_backbone}_{head_type}"


def _build_experiment_metadata(cfg, *, epochs: int, batch_size: int, pretrained: bool,
                               topk: int, seed_for_train: int, lr: float,
                               weight_decay: float, scheduler_factor: float,
                               scheduler_patience: int):
    model_cfg = cfg["dataset"]["model"]
    lowlight_cfg = cfg["dataset"].get("augmentation", {}).get("lowlight", {})
    return {
        "model_name": model_cfg["name"],
        "rgb_backbone": model_cfg.get("rgb_backbone", ""),
        "depth_backbone": model_cfg.get("depth_backbone", ""),
        "head_type": model_cfg.get("head_type", "transformer"),
        "fusion_dim": model_cfg.get("fusion_dim", ""),
        "fusion_grid_size": model_cfg.get("fusion_grid_size", ""),
        "pretrained": pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "topk": topk,
        "lr": lr,
        "weight_decay": weight_decay,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "split_seed": cfg["dataset"]["split"]["seed"],
        "train_seed": seed_for_train,
        "lowlight_train": lowlight_cfg.get("enable_train", ""),
        "lowlight_eval": lowlight_cfg.get("enable_eval", ""),
        "lowlight_p": lowlight_cfg.get("p", ""),
    }


def train_midfusion(cfg, epochs: int = 40, batch_size: int = 32,
                    pretrained: bool = True, topk: int = 10,
                    seed_for_train: int = 42, lr: float = 1e-4,
                    weight_decay: float = 5e-4,
                    scheduler_factor: float = 0.5,
                    scheduler_patience: int = 2,
                    fusion_dim: int | None = None,
                    fusion_grid_size: int | None = None,
                    head_type: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mid-fusion pipeline is only for RGB-D.
    mode = "rgbd"

    if fusion_dim is not None:
        cfg["dataset"]["model"]["fusion_dim"] = fusion_dim
    if fusion_grid_size is not None:
        cfg["dataset"]["model"]["fusion_grid_size"] = fusion_grid_size
    if head_type is not None:
        cfg["dataset"]["model"]["head_type"] = head_type

    ds_train = SUNRGBDObjectROIDatasetMidFusionLowLight(cfg, split="train", mode=mode, topk=topk)
    ds_val = SUNRGBDObjectROIDatasetMidFusionLowLight(cfg, split="val", mode=mode, label2id=ds_train.label2id)
    ds_test = SUNRGBDObjectROIDatasetMidFusionLowLight(cfg, split="test", mode=mode, label2id=ds_train.label2id)

    _filter_keep_classes(ds_train, ds_val, ds_test)

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    num_classes = len(ds_train.label2id)
    # Keep in_channels=4 only for compatibility with your existing build_model signature.
    model = build_model(cfg, num_classes=num_classes, in_channels=4, pretrained=pretrained).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-7,
    )

    best_val_acc = -1.0
    best_epoch = -1
    patience = 8
    bad_epochs = 0
    min_delta = 1e-3

    model_name = cfg["dataset"]["model"]["name"]
    experiment_tag = _midfusion_experiment_tag(cfg)
    save_path_model = f'../checkpoints/object_best_rgbd_seed_{seed_for_train}_model_{model_name}_{experiment_tag}_lowlight.pth'
    save_path_outcome = f'../outcomes/object_outcomes_{model_name}_{experiment_tag}_lowlight.csv'
    logger = ExperimentLogger(save_path_outcome)
    experiment_metadata = _build_experiment_metadata(
        cfg,
        epochs=epochs,
        batch_size=batch_size,
        pretrained=pretrained,
        topk=topk,
        seed_for_train=seed_for_train,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
    )

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_midfusion(model, loader_train, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch_midfusion(model, loader_val, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        scheduler.step(val_loss)

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                save_path_model,
            )
        else:
            bad_epochs += 1

        print(bad_epochs)
        if bad_epochs >= patience:
            break

    _, test_acc = evaluate_for_object_midfusion(cfg=cfg, modelPath=save_path_model, test=ds_test)
    logger.log(
        mode="rgbd",
        seed=seed_for_train,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        test_acc=test_acc,
        **experiment_metadata,
    )


if __name__ == "__main__":
    base_cfg = load_yaml("./configs/dataset_sun_rgbd_object.yaml")
    preset_names = [
        "untuned_adapt",
    ]
    for preset_name in preset_names:
        cfg = copy.deepcopy(base_cfg)
        preset_train_kwargs = apply_experiment_preset(cfg, preset_name)
        print(f"Running low-light preset: {preset_name}")
        print(
            "model=",
            cfg["dataset"]["model"]["name"],
            "rgb_backbone=",
            cfg["dataset"]["model"]["rgb_backbone"],
            "depth_backbone=",
            cfg["dataset"]["model"]["depth_backbone"],
            "head_type=",
            cfg["dataset"]["model"]["head_type"],
            "lowlight_train=",
            cfg["dataset"]["augmentation"]["lowlight"]["enable_train"],
            "lowlight_eval=",
            cfg["dataset"]["augmentation"]["lowlight"]["enable_eval"],
        )

        training_seeds = [42, 123, 3407]
        for seed in training_seeds:
            set_global_seed(seed)
            train_midfusion(
                cfg,
                epochs=40,
                batch_size=32,
                pretrained=True,
                topk=10,
                seed_for_train=seed,
                **preset_train_kwargs,
            )
