from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter


def _clamp_box(xmin, ymin, xmax, ymax, w, h):
    xmin = int(max(0, min(xmin, w - 1)))
    xmax = int(max(0, min(xmax, w - 1)))
    ymin = int(max(0, min(ymin, h - 1)))
    ymax = int(max(0, min(ymax, h - 1)))
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


def _norm_minmax(depth_np: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    minimum = float(np.min(depth_np))
    maximum = float(np.max(depth_np))
    if maximum - minimum < eps:
        return np.zeros_like(depth_np, dtype=np.float32)
    return (depth_np - minimum) / (maximum - minimum)


class SUNRGBDObjectROIDatasetMidFusion(Dataset):
    """ROI object classification dataset for dual-branch RGB-D models.

    Unlike the early-fusion dataset, this class keeps RGB and depth separate.

    Returns:
        - mode='rgb'   -> x_rgb, y
        - mode='depth' -> x_depth, y
        - mode='rgbd'  -> (x_rgb, x_depth), y
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train",
        mode: str = "rgbd",
        label2id: Dict[str, int] | None = None,
        topk: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.mode = mode  # 'rgb', 'depth', 'rgbd'

        ds_cfg = cfg["dataset"]
        assert ds_cfg.get("task") == "object_roi", (
            f"Dataset task must be object_roi, got {ds_cfg.get('task')}"
        )

        self.root = Path(ds_cfg["root"])
        self.sensor = ds_cfg["sensor"]
        self.subset = ds_cfg["subset"]

        self.img_size = tuple(ds_cfg["preprocessing"]["image_size"])
        self.min_box = int(ds_cfg.get("roi", {}).get("min_box_size", 20))
        self.drop_unknown = bool(ds_cfg.get("roi", {}).get("drop_unknown", True))

        if split == "train":
            self.roi_index_file = Path(ds_cfg["roi"]["index_train"])
        elif split == "val":
            self.roi_index_file = Path(ds_cfg["roi"]["index_val"])
        elif split == "test":
            self.roi_index_file = Path(ds_cfg["roi"]["index_test"])
        else:
            raise ValueError(f"Unknown split: {split}")

        depth_cfg = ds_cfg["preprocessing"].get("depth", {})
        self.depth_fill = depth_cfg.get("fill_missing", 0)
        self.depth_norm = depth_cfg.get("normalize", "minmax")
        self.depth_to_3ch = bool(depth_cfg.get("to_3ch", False))
        if self.mode == "rgbd" and self.depth_to_3ch:
            raise ValueError(
                "For dual-branch mid-fusion, depth_to_3ch must be False so the depth branch remains 1-channel."
            )

        all_samples = self._load_jsonl(self.roi_index_file)

        if self.drop_unknown:
            all_samples = [s for s in all_samples if s.get("label") not in (None, "unknown")]

        self.keep_labels = None
        if split == "train" and topk is not None:
            cnt = Counter(s["label"] for s in all_samples)
            self.keep_labels = {lab for lab, _ in cnt.most_common(int(topk))}
            all_samples = [s for s in all_samples if s["label"] in self.keep_labels]

        if label2id is None:
            self.label2id = self._build_label_map(all_samples)
        else:
            self.label2id = label2id

        self.label_to_index = self.label2id
        self.samples = [s for s in all_samples if s.get("label") in self.label2id]

        self.rgb_transform_train = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        ])
        self.rgb_transform_eval = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.depth_resize = transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def _build_label_map(self, samples: List[Dict[str, Any]]) -> Dict[str, int]:
        labels = []
        for s in samples:
            lab = s.get("label", "unknown")
            if self.drop_unknown and (lab is None or lab == "unknown"):
                continue
            labels.append(lab)
        labels = sorted(set(labels))
        return {lab: i for i, lab in enumerate(labels)}

    def __len__(self):
        return len(self.samples)

    def _load_rgb_crop(self, s: Dict[str, Any]) -> Image.Image:
        img = Image.open(s["image_path"]).convert("RGB")
        w, h = img.size
        xmin, ymin, xmax, ymax = _clamp_box(s["xmin"], s["ymin"], s["xmax"], s["ymax"], w, h)
        if (xmax - xmin) < self.min_box or (ymax - ymin) < self.min_box:
            return img.crop((0, 0, min(self.min_box, w), min(self.min_box, h)))
        return img.crop((xmin, ymin, xmax, ymax))

    def _load_depth_crop(self, s: Dict[str, Any]) -> torch.Tensor:
        depth_path = s.get("depth_path")
        if depth_path is None:
            image_path = Path(s["image_path"])
            depth_path = str(image_path.parent.parent / "depth" / (image_path.stem + ".png"))

        d = Image.open(depth_path)
        w, h = d.size
        xmin, ymin, xmax, ymax = _clamp_box(s["xmin"], s["ymin"], s["xmax"], s["ymax"], w, h)
        d = d.crop((xmin, ymin, xmax, ymax))
        d = self.depth_resize(d)

        d_np = np.array(d).astype(np.float32)
        if self.depth_fill is not None:
            pass

        if self.depth_norm == "minmax":
            d_np = _norm_minmax(d_np)
        elif self.depth_norm == "zscore":
            mu, sigma = float(d_np.mean()), float(d_np.std() + 1e-6)
            d_np = (d_np - mu) / sigma

        d_t = torch.from_numpy(d_np)[None, ...]
        if self.depth_to_3ch:
            d_t = d_t.repeat(3, 1, 1)
        return d_t

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        label = s.get("label", "unknown")
        if self.drop_unknown and (label is None or label == "unknown"):
            label = next(iter(self.label2id.keys()))
        y = self.label2id[label]

        rgb_crop = self._load_rgb_crop(s)
        if self.split == "train":
            x_rgb = self.rgb_transform_train(rgb_crop)
        else:
            x_rgb = self.rgb_transform_eval(rgb_crop)

        if self.mode == "rgb":
            return x_rgb, y

        x_d = self._load_depth_crop(s)
        if self.mode == "depth":
            return x_d, y

        if x_d.shape[0] != 1:
            raise ValueError(
                f"Expected depth tensor to have 1 channel for mid-fusion, got shape {tuple(x_d.shape)}"
            )
        return (x_rgb, x_d), y
