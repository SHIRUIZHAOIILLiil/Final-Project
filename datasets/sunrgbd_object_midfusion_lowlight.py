from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import random
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
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


class RandomLowLightRGB:
    """Apply synthetic low-light degradation to RGB only.

    This is intended for RGB-D experiments where depth stays unchanged.
    """

    def __init__(
        self,
        p: float = 0.7,
        brightness_range=(0.25, 0.7),
        contrast_range=(0.6, 0.95),
        gamma_range=(1.2, 2.4),
        noise_std_range=(0.01, 0.05),
        blur_radius_range=(0.0, 1.2),
    ):
        self.p = float(p)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std_range = noise_std_range
        self.blur_radius_range = blur_radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        # darker illumination
        brightness = random.uniform(*self.brightness_range)
        img = TF.adjust_brightness(img, brightness)

        # flatter contrast, common in dim scenes
        contrast = random.uniform(*self.contrast_range)
        img = TF.adjust_contrast(img, contrast)

        # gamma > 1 darkens shadows/mid-tones further
        gamma = random.uniform(*self.gamma_range)
        img = TF.adjust_gamma(img, gamma)

        # optional blur
        blur_radius = random.uniform(*self.blur_radius_range)
        if blur_radius > 1e-6:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # sensor noise in low-light
        arr = np.asarray(img).astype(np.float32) / 255.0
        noise_std = random.uniform(*self.noise_std_range)
        arr = np.clip(arr + np.random.normal(0.0, noise_std, arr.shape), 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)


class SUNRGBDObjectROIDatasetMidFusionLowLight(Dataset):
    """ROI object classification dataset for dual-branch RGB-D models.

    Returns:
        - mode='rgb'   -> x_rgb, y
        - mode='depth' -> x_depth, y
        - mode='rgbd'  -> (x_rgb, x_depth), y

    Difference from the standard mid-fusion dataset:
        training RGB crops can be synthetically degraded into low-light variants
        while depth remains unchanged.
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

        lowlight_cfg = ds_cfg.get("augmentation", {}).get("lowlight", {})
        self.lowlight_enable_train = bool(lowlight_cfg.get("enable_train", True))
        self.lowlight_enable_eval = bool(lowlight_cfg.get("enable_eval", False))
        self.lowlight_aug = RandomLowLightRGB(
            p=float(lowlight_cfg.get("p", 0.7)),
            brightness_range=tuple(lowlight_cfg.get("brightness_range", [0.25, 0.7])),
            contrast_range=tuple(lowlight_cfg.get("contrast_range", [0.6, 0.95])),
            gamma_range=tuple(lowlight_cfg.get("gamma_range", [1.2, 2.4])),
            noise_std_range=tuple(lowlight_cfg.get("noise_std_range", [0.01, 0.05])),
            blur_radius_range=tuple(lowlight_cfg.get("blur_radius_range", [0.0, 1.2])),
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

        # geometric augmentation first, tensor conversion last
        self.rgb_resize = transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.rgb_base_train = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        ])
        self.rgb_base_eval = transforms.Compose([
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

    def _transform_rgb(self, rgb_crop: Image.Image) -> torch.Tensor:
        rgb_crop = self.rgb_resize(rgb_crop)

        apply_lowlight = (
            (self.split == "train" and self.lowlight_enable_train)
            or (self.split != "train" and self.lowlight_enable_eval)
        )
        if apply_lowlight:
            rgb_crop = self.lowlight_aug(rgb_crop)

        if self.split == "train":
            return self.rgb_base_train(rgb_crop)
        return self.rgb_base_eval(rgb_crop)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        label = s.get("label", "unknown")
        if self.drop_unknown and (label is None or label == "unknown"):
            label = next(iter(self.label2id.keys()))
        y = self.label2id[label]

        rgb_crop = self._load_rgb_crop(s)
        x_rgb = self._transform_rgb(rgb_crop)

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
