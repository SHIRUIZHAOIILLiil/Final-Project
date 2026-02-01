from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.transforms import InterpolationMode

def _read_single_token(path: Path, lowercase: bool = True) -> str:
    s = path.read_text(encoding="utf-8", errors="ignore").strip()
    s = s.splitlines()[0].strip()
    return s.lower() if lowercase else s


def _pil_rgb_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    arr = np.transpose(arr, (2, 0, 1))               # (3,H,W)
    return torch.from_numpy(arr)


def _load_depth_png(path: Path) -> np.ndarray:
    d = np.array(Image.open(path)).astype(np.float32)
    return d


def _normalize_depth(depth: np.ndarray, mode: str = "minmax", fill_missing: float = 0.0) -> np.ndarray:
    d = depth.copy()
    invalid = ~np.isfinite(d) | (d <= 0)
    d[invalid] = fill_missing

    if mode in (None, "none"):
        return d

    valid = np.isfinite(d) & (d > fill_missing)
    if valid.any():
        vmin = float(d[valid].min())
        vmax = float(d[valid].max())
        if vmax > vmin:
            d = (d - vmin) / (vmax - vmin)
        else:
            d = d * 0.0
    else:
        d = d * 0.0

    return np.clip(d, 0.0, 1.0)

def _build_index(
    base: Path,
    rgb_cfg: dict,
    depth_cfg: dict,
    label_filename: str,
):
    if not base.exists():
        raise FileNotFoundError(f"Dataset base not found: {base}")

    # collect valid samples
    items: List[Tuple[str, Path, Path, Path]] = []
    for sd in sorted([p for p in base.iterdir() if p.is_dir()]):
        sid = sd.name
        rgb_path = sd / rgb_cfg["dir"] / rgb_cfg.get("pattern", "{id}.jpg").format(id=sid)
        depth_path = sd / depth_cfg["dir"] / depth_cfg.get("pattern", "{id}.png").format(id=sid)
        label_path = sd / label_filename
        if rgb_path.exists() and depth_path.exists() and label_path.exists():
            items.append((sid, rgb_path, depth_path, label_path))

    if not items:
        raise RuntimeError(f"No valid samples found under: {base}")
    return items

def _make_split(
        labels: List[str],
        label_to_index: Dict[str, int],
        tr: float,
        va: float,
        seed: int,
        stratify: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    if stratify:
        by_class: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            c = label_to_index[lab]
            by_class.setdefault(c, []).append(i)
        for c in by_class:
            rng.shuffle(by_class[c])

        train_idx, val_idx, test_idx = [], [], []
        for c, idxs in by_class.items():
            n = len(idxs)
            n_tr = int(round(n * tr))
            n_va = int(round(n * va))
            n_tr = min(n_tr, n)
            n_va = min(n_va, n - n_tr)
            train_idx += idxs[:n_tr]
            val_idx += idxs[n_tr:n_tr + n_va]
            test_idx += idxs[n_tr + n_va:]

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
    else:
        all_idx = list(range(len(labels)))
        rng.shuffle(all_idx)

        n = len(all_idx)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))

        train_idx = all_idx[:n_tr]
        val_idx = all_idx[n_tr:n_tr + n_va]
        test_idx = all_idx[n_tr + n_va:]

    return train_idx, val_idx, test_idx


def _random_erasing_sync(rgb, depth, p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0):
    """
    rgb:   Tensor (3, H, W)
    depth: Tensor (1, H, W)
    """

    if random.random() > p:
        return rgb, depth

    _, H, W = rgb.shape
    area = H * W

    for _ in range(10):  # 最多尝试 10 次，和 torchvision 一致
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if h < H and w < W:
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)

            rgb[:, top:top+h, left:left+w] = value
            depth[:, top:top+h, left:left+w] = value
            return rgb, depth

    return rgb, depth

class SUNRGBDSceneDataset(Dataset):
    """
    YAML-driven SUNRGBD NYUdata (scene-level) dataset.
    Expected structure:
      <root>/<sensor>/<subset>/<ID>/
        image/<ID>.jpg
        depth/<ID>.png
        scene.txt   (single token label)
    """

    def __init__(
        self,
        cfg: dict,
        split: str = "train",
        label_to_index: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}
        self.cfg = cfg
        self.split = split

        dcfg = self.cfg["dataset"]
        root = Path(dcfg["root"])
        sensor = dcfg["sensor"]
        subset = dcfg["subset"]

        rgb_cfg = dcfg["rgb"]
        depth_cfg = dcfg["depth"]

        task = dcfg["task"]
        if task != "image_pair":
            raise ValueError(f"Current dataset supports task='image_pair' only, got: {task}")

        # preprocessing
        pcfg = dcfg["preprocessing"]
        img_h, img_w = pcfg["image_size"]

        depth_pcfg = pcfg["depth"]
        depth_fill = float(depth_pcfg["fill_missing"])
        depth_norm = depth_pcfg["normalize"]

        # labels (per-sample)
        lcfg = dcfg["labels"]
        if lcfg.get("type") != "per_single_file":
            raise ValueError("labels.type must be 'per_single_file' for this dataset.")
        label_filename = lcfg["filename"]
        lowercase = bool(lcfg["lowercase"])

        # split config
        scfg = dcfg["split"]
        method = scfg["method"]
        if method != "random":
            raise ValueError("Only split.method='random' is supported in this minimal baseline.")
        tr = float(scfg["train"])
        va = float(scfg["val"])
        te = float(scfg["test"])
        seed = int(scfg["seed"])

        base = root / sensor / subset
        items = _build_index(base, rgb_cfg, depth_cfg, label_filename)
        self.__all_items = items
        # read labels & build mapping
        all_labels = [_read_single_token(lp, lowercase=lowercase) for _, _, _, lp in items]
        if label_to_index is None:
            uniq = sorted(set(all_labels))
            self.label_to_index = {lab: i for i, lab in enumerate(uniq)}
        else:
            self.label_to_index = dict(label_to_index)

        # stratified-ish random split by class

        train_idx, val_idx, test_idx = _make_split(all_labels, self.label_to_index, tr, va, seed)
        self._split_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
        use_idx = self._split_indices[split]

        self.items = [items[i] for i in use_idx]

        # store processing params
        self._img_size = (img_w, img_h)  # PIL wants (W,H)

        # Data augmentation
        if split == "train":
            self.crop_scale = (0.8, 1.0)
            self.flip_p = 0.5

            self.rgb_color = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        else:
            self.crop_scale = None
            self.flip_p = 0.0
            self.rgb_color = None
            self.rgb_tensor_transform = None

        self._depth_fill = depth_fill
        self._depth_norm = depth_norm
        self._lowercase = lowercase
        self._label_filename = label_filename

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sid, rgb_path, depth_path, label_path = self.items[idx]

        # label
        label_str = _read_single_token(label_path, lowercase=self._lowercase)
        y = self.label_to_index[label_str]

        # rgb
        rgb = Image.open(rgb_path).convert("RGB")

        # depth
        depth = _load_depth_png(depth_path)
        depth_img = Image.fromarray(depth)
        if self.split == "train":
            # aspect ratio = W / H from 0.75 to 1.333
            # 0.75 裁剪框偏高, 1.33 裁剪框偏宽
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                rgb, scale=self.crop_scale, ratio=(3 / 4, 4 / 3)
            )
            rgb = functional.resized_crop(
                # 直接把整个图缩放
                rgb, i, j, h, w, size=self._img_size, interpolation=InterpolationMode.BILINEAR #缩放采用的线性插值
            )
            depth_img = functional.resized_crop(
                depth_img, i, j, h, w, size=self._img_size, interpolation=InterpolationMode.NEAREST
            )
            # Sample flip ONCE and apply to both
            if random.random() < self.flip_p:
                rgb = functional.hflip(rgb)
                depth_img = functional.hflip(depth_img)

            # RGB-only color jitter (after geometry)
            if self.rgb_color is not None:
                rgb = self.rgb_color(rgb)
        else:
            # val/test deterministic resize
            rgb = functional.resize(rgb, self._img_size, interpolation=InterpolationMode.BILINEAR)
            depth_img = functional.resize(depth_img, self._img_size, interpolation=InterpolationMode.NEAREST)



        # rgb to tensor
        rgb_t = _pil_rgb_to_tensor(rgb)  # (3,H,W)

        depth = np.array(depth_img).astype(np.float32)
        depth = _normalize_depth(depth, mode=self._depth_norm, fill_missing=self._depth_fill)
        depth = depth[:, :, None]  # (H,W,1)
        depth_t = torch.from_numpy(np.transpose(depth, (2, 0, 1))).float()  # (1,H,W)

        rgb_t, depth_t = _random_erasing_sync(
            rgb_t,
            depth_t,
            p=0.25,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0
        )

        return rgb_t, depth_t, torch.tensor(y, dtype=torch.long), sid

    def summarize(self) -> None:
        for k, v in self._split_indices.items():
            print(f"split: {k}")
            print(f"Total Samples: {len(self._split_indices[k])}")

            label_count = {}
            for i in v:
                _, _, _, label_path = self.__all_items[i]
                label = _read_single_token(label_path, lowercase=self._lowercase)

                if label not in label_count:
                    label_count[label] = 0
                label_count[label] += 1

            for label, count in sorted(label_count.items()):
                print(f"  {label}: {count}")
            print()


