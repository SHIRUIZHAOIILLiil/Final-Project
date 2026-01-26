from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


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

        dcfg = cfg["dataset"]
        root = Path(dcfg["root"])
        sensor = dcfg.get("sensor", "kv1")
        subset = dcfg.get("subset", "NYUdata")

        rgb_cfg = dcfg["rgb"]
        depth_cfg = dcfg["depth"]

        task = dcfg.get("task", "image_pair")
        if task != "image_pair":
            raise ValueError(f"Current dataset supports task='image_pair' only, got: {task}")

        # preprocessing
        pcfg = dcfg.get("preprocessing", {})
        img_h, img_w = pcfg.get("image_size", [224, 224])

        depth_pcfg = pcfg.get("depth", {})
        depth_fill = float(depth_pcfg.get("fill_missing", 0.0))
        depth_norm = depth_pcfg.get("normalize", "minmax")

        # labels (per-sample)
        lcfg = dcfg.get("labels", {})
        if lcfg.get("type") != "per_single_file":
            raise ValueError("labels.type must be 'per_single_file' for this dataset.")
        label_filename = lcfg.get("filename", "scene.txt")
        lowercase = bool(lcfg.get("lowercase", True))

        # split config
        scfg = dcfg.get("split", {})
        method = scfg.get("method", "random")
        if method != "random":
            raise ValueError("Only split.method='random' is supported in this minimal baseline.")
        tr = float(scfg.get("train", 0.7))
        va = float(scfg.get("val", 0.15))
        te = float(scfg.get("test", 0.15))
        seed = int(scfg.get("seed", 42))

        base = root / sensor / subset
        if not base.exists():
            raise FileNotFoundError(f"Dataset base not found: {base}")

        # collect valid samples
        items: List[Tuple[str, Path, Path, Path]] = []
        for sd in sorted([p for p in base.iterdir() if p.is_dir()]):
            sid = sd.name
            rgb_path = sd / rgb_cfg.get("dir", "image") / rgb_cfg.get("pattern", "{id}.jpg").format(id=sid)
            depth_path = sd / depth_cfg.get("dir", "depth") / depth_cfg.get("pattern", "{id}.png").format(id=sid)
            label_path = sd / label_filename
            if rgb_path.exists() and depth_path.exists() and label_path.exists():
                items.append((sid, rgb_path, depth_path, label_path))

        if not items:
            raise RuntimeError(f"No valid samples found under: {base}")

        # read labels & build mapping
        all_labels = [_read_single_token(lp, lowercase=lowercase) for _, _, _, lp in items]
        if label_to_index is None:
            uniq = sorted(set(all_labels))
            self.label_to_index = {lab: i for i, lab in enumerate(uniq)}
        else:
            self.label_to_index = dict(label_to_index)

        # stratified-ish random split by class
        rng = np.random.default_rng(seed)
        by_class: Dict[int, List[int]] = {}
        for i, lab in enumerate(all_labels):
            c = self.label_to_index[lab]
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

        use_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
        self.items = [items[i] for i in use_idx]

        # store processing params
        self._img_size = (img_w, img_h)  # PIL wants (W,H)
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
        rgb = Image.open(rgb_path).convert("RGB").resize(self._img_size, resample=Image.BILINEAR)
        rgb_t = _pil_rgb_to_tensor(rgb)  # (3,H,W)

        # depth
        depth = _load_depth_png(depth_path)
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize(self._img_size, resample=Image.NEAREST)
        depth = np.array(depth_img).astype(np.float32)

        depth = _normalize_depth(depth, mode=self._depth_norm, fill_missing=self._depth_fill)
        depth = depth[:, :, None]  # (H,W,1)
        depth_t = torch.from_numpy(np.transpose(depth, (2, 0, 1))).float()  # (1,H,W)

        return rgb_t, depth_t, torch.tensor(y, dtype=torch.long), sid
