from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image

import json, random

STUFF_NAMES = {"wall", "floor", "ceiling", "window", "counter", "blinds", "door",}

def valid_polygon(xs: List[int], ys: List[int]) -> bool:
    return xs is not None and ys is not None and len(xs) >= 3 and len(xs) == len(ys)


def bbox_from_polygon(xs: List[int], ys: List[int]) -> Tuple[int, int, int, int]:
    return min(xs), min(ys), max(xs), max(ys)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def keep_roi(label: str, xmin: int, ymin: int, xmax: int, ymax: int, W: int, H: int) -> bool:
    w = max(1, xmax - xmin)
    h = max(1, ymax - ymin)
    area_ratio = (w * h) / float(W * H)

    # 1) 明确剔除
    if label in STUFF_NAMES:
        return False

    # 2) 超大块剔除：典型 wall/floor
    if area_ratio >= 0.60:
        return False

    # 3) 贴边长条剔除：典型 ceiling strip / wall edge
    touches_border = (xmin <= 1) or (ymin <= 1) or (xmax >= W - 2) or (ymax >= H - 2)
    aspect = max(w / h, h / w)
    if touches_border and aspect >= 8 and area_ratio >= 0.05:
        return False

    return True

def main(seed:int = 42):
    nyu_root = Path(r"E:/dataset/SUNRGBD/kv1/NYUdata")  # contains NYU0001, NYU0002, ...
    out_dir = Path(r"E:/dataset/SUNRGBD/roi_index")

    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    seed = seed

    min_box_size = 20
    drop_unknown = True
    clamp_negative = True

    rng = random.Random(seed)

    rows_by_image: Dict[str, List[Dict[str, Any]]] = {}
    n_folders = 0
    n_missing_index = 0
    n_total_polys = 0
    n_kept = 0
    n_skipped = 0

    for sample_dir in sorted([p for p in nyu_root.iterdir() if p.is_dir() and p.name.startswith("NYU")]):
        n_folders += 1
        image_id = sample_dir.name

        ann_index = sample_dir / "annotation2Dfinal" / "index.json"
        if not ann_index.exists():
            n_missing_index += 1
            continue

        rgb_path = sample_dir / "image" / f"{image_id}.jpg"
        depth_path = sample_dir / "depth" / f"{image_id}.png"


        if not rgb_path.exists():
            # skip if image missing
            n_skipped += 1
            continue
        with Image.open(rgb_path) as im:
            W, H = im.size

        item = json.loads(ann_index.read_text(encoding="utf-8"))

        # build objects index->name
        obj_list = item.get("objects") or []

        def idx2name(idx: int) -> str:
            if idx < 0 or idx >= len(obj_list):
                return "unknown"
            ent = obj_list[idx]
            if ent is None:
                return "unknown"
            return (ent.get("name") or "unknown").strip().lower()

        frames = item.get("frames") or []
        if not frames:
            continue
        polygons = frames[0].get("polygon") or []

        for poly in polygons:
            n_total_polys += 1
            xs = poly.get("x") or []
            ys = poly.get("y") or []
            obj_idx = int(poly.get("object", -1))
            label = idx2name(obj_idx)

            # filter
            if not valid_polygon(xs, ys):
                n_skipped += 1
                continue
            if drop_unknown and label == "unknown":
                n_skipped += 1
                continue

            # clamp negatives (your earlier example had -1)
            if clamp_negative:
                xs = [max(0, int(x)) for x in xs]
                ys = [max(0, int(y)) for y in ys]

            xmin, ymin, xmax, ymax = bbox_from_polygon(xs, ys)

            if (xmax - xmin) < min_box_size or (ymax - ymin) < min_box_size:
                n_skipped += 1
                continue
            if not keep_roi(label, xmin, ymin, xmax, ymax, W, H):
                n_skipped += 1
                continue

            xmin = max(0, min(xmin, W - 1))
            xmax = max(0, min(xmax, W - 1))
            ymin = max(0, min(ymin, H - 1))
            ymax = max(0, min(ymax, H - 1))

            row = {
                "scene_id": image_id,  # for by_image split
                "image_id": image_id,
                "image_path": str(rgb_path),
                "depth_path": str(depth_path) if depth_path.exists() else None,
                "img_w": int(W),
                "img_h": int(H),
                "xmin": int(xmin),
                "ymin": int(ymin),
                "xmax": int(xmax),
                "ymax": int(ymax),
                "label": label,
            }
            rows_by_image.setdefault(image_id, []).append(row)
            n_kept += 1

        # ===== split by image_id to avoid leakage =====
    image_ids = list(rows_by_image.keys())
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_tr = int(round(n * train_ratio))
    n_va = int(round(n * val_ratio))

    tr_ids = set(image_ids[:n_tr])
    va_ids = set(image_ids[n_tr:n_tr + n_va])
    te_ids = set(image_ids[n_tr + n_va:])

    rows_train, rows_val, rows_test = [], [], []
    for iid, rows in rows_by_image.items():
        if iid in tr_ids:
            rows_train.extend(rows)
        elif iid in va_ids:
            rows_val.extend(rows)
        else:
            rows_test.extend(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "roi_train.jsonl", rows_train)
    write_jsonl(out_dir / "roi_val.jsonl", rows_val)
    write_jsonl(out_dir / "roi_test.jsonl", rows_test)
    write_jsonl(out_dir / "roi_all.jsonl", rows_train + rows_val + rows_test)

    print("Done.")
    print(f"Sample folders scanned: {n_folders}")
    print(f"Missing annotation2final/index.json: {n_missing_index}")
    print(f"Images used: {len(rows_by_image)}")
    print(f"Total polygons: {n_total_polys}")
    print(f"Kept ROIs: {n_kept}")
    print(f"Skipped polygons: {n_skipped}")
    print(f"Train/Val/Test images: {len(tr_ids)}/{len(va_ids)}/{len(te_ids)}")
    print(f"Train/Val/Test ROIs: {len(rows_train)}/{len(rows_val)}/{len(rows_test)}")
    print(f"Output dir: {out_dir}")

if __name__ == "__main__":
    main(seed=42)