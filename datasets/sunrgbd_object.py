from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import random, json
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
    # ensure proper ordering
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


def _norm_minmax(depth_np: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # depth_np: float32
    minimum = float(np.min(depth_np))
    maximum = float(np.max(depth_np))
    if maximum - minimum < eps:
        return np.zeros_like(depth_np, dtype=np.float32)
    return (depth_np - minimum) / (maximum - minimum)


class SUNRGBDObjectROIDataset(Dataset):
    """
    ROI object classification dataset.

    Expects roi_samples.jsonl where each line is like:
    {
      "image_id": "NYU0001",
      "image_path": "E:/dataset/SUNRGBD/kv1/NYUdata/NYU0001/image/NYU0001.jpg",
      "depth_path": "E:/dataset/SUNRGBD/kv1/NYUdata/NYU0001/depth/NYU0001.png",   (optional)
      "xmin": 376, "ymin": 269, "xmax": 459, "ymax": 427,
      "label": "cabinet",
      "scene_id": "NYU0001"  (optional but recommended for by_image split)
    }
    """

    def __init__(self, cfg: Dict[str, Any], split: str = "train", mode: str = "rgb",
                 label2id: Dict[str, int] | None = None, topk: int | None = None
                 ):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.mode = mode  # "rgb" or "rgbd"

        ds_cfg = cfg["dataset"]
        assert ds_cfg.get("task") == "object_roi", f"Dataset task must be object_roi, got {ds_cfg.get('task')}"

        self.root = Path(ds_cfg["root"])
        self.sensor = ds_cfg["sensor"]
        self.subset = ds_cfg["subset"]

        self.img_size = tuple(ds_cfg["preprocessing"]["image_size"])  # (H, W)
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

        # Depth preprocessing
        depth_cfg = ds_cfg["preprocessing"].get("depth", {})
        self.depth_fill = depth_cfg.get("fill_missing", 0)
        self.depth_norm = depth_cfg.get("normalize", "minmax")
        self.depth_to_3ch = bool(depth_cfg.get("to_3ch", False))

        # Build samples + label map
        all_samples = self._load_jsonl(self.roi_index_file)

        # 1) drop unknown early
        if self.drop_unknown:
            all_samples = [s for s in all_samples if s.get("label") not in (None, "unknown")]

        # 2) topK filtering ONLY on train (recommended)
        self.keep_labels = None
        if split == "train" and topk is not None:
            cnt = Counter(s["label"] for s in all_samples)
            self.keep_labels = {lab for lab, _ in cnt.most_common(int(topk))}
            all_samples = [s for s in all_samples if s["label"] in self.keep_labels]

        # 3) label mapping: reuse shared map if provided, else build from (filtered) train
        if label2id is None:
            self.label2id = self._build_label_map(all_samples)  # or build from sorted set
        else:
            self.label2id = label2id

        # optional alias for your training code compatibility
        self.label_to_index = self.label2id

        # 4) filter val/test (and even train) to labels in label2id
        # this removes "val-only labels" cleanly
        self.samples = [s for s in all_samples if s.get("label") in self.label2id]

        # Transforms: ROI already cropped, so just resize + light aug
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

        # Depth: treat as single-channel tensor
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
        # filter too small
        if (xmax - xmin) < self.min_box or (ymax - ymin) < self.min_box:
            # return a minimal crop; caller may retry by picking another sample if desired
            return img.crop((0, 0, min(self.min_box, w), min(self.min_box, h)))
        return img.crop((xmin, ymin, xmax, ymax))

    def _load_depth_crop(self, s: Dict[str, Any]) -> torch.Tensor:
        depth_path = s.get("depth_path")
        if depth_path is None:
            # build default path if not provided
            # E:/dataset/SUNRGBD/kv1/NYUdata/NYU0001/depth/NYU0001.png
            image_path = Path(s["image_path"])
            depth_path = str(image_path.parent.parent / "depth" / (image_path.stem + ".png"))

        d = Image.open(depth_path)  # usually 16-bit PNG
        # crop using same bbox (PIL uses same coordinate convention)
        w, h = d.size
        xmin, ymin, xmax, ymax = _clamp_box(s["xmin"], s["ymin"], s["xmax"], s["ymax"], w, h)
        d = d.crop((xmin, ymin, xmax, ymax))
        d = self.depth_resize(d)

        d_np = np.array(d).astype(np.float32)
        # fill missing (0s often mean missing, but depends on dataset)
        if self.depth_fill is not None:
            # if fill_missing=0, this is no-op; if you want treat 0 as missing, you'd need a mask rule
            pass

        if self.depth_norm == "minmax":
            d_np = _norm_minmax(d_np)
        elif self.depth_norm == "zscore":
            mu, sigma = float(d_np.mean()), float(d_np.std() + 1e-6)
            d_np = (d_np - mu) / sigma

        d_t = torch.from_numpy(d_np)[None, ...]  # (1,H,W)
        if self.depth_to_3ch:
            d_t = d_t.repeat(3, 1, 1)
        return d_t

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        label = s.get("label", "unknown")
        if self.drop_unknown and (label is None or label == "unknown"):
            # should be filtered already, but keep safe
            label = next(iter(self.label2id.keys()))
        y = self.label2id[label]

        rgb_crop = self._load_rgb_crop(s)

        if self.split == "train":
            x_rgb = self.rgb_transform_train(rgb_crop)
        else:
            x_rgb = self.rgb_transform_eval(rgb_crop)

        if self.mode == "rgb":
            return x_rgb, y
        elif self.mode == "depth":
            x_d = self._load_depth_crop(s)
            return x_d, y

        # rgbd
        x_d = self._load_depth_crop(s)
        return (x_rgb, x_d), y