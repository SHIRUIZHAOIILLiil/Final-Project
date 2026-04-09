import csv
import os
import yaml, torch
from pathlib import Path

def load_yaml(path: str) -> dict:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_input(rgb, depth, mode: str):
    if mode == "rgb":
        return rgb
    elif mode == "depth":
        return depth
    elif mode == "rgbd":
        return torch.cat([rgb, depth], dim=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

class ExperimentLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.fieldnames = []
        if os.path.exists(filepath):
            with open(filepath, "r", newline="") as f:
                reader = csv.DictReader(f)
                self.fieldnames = list(reader.fieldnames or [])

    def _read_existing_rows(self):
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _rewrite_with_fieldnames(self, rows):
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in self.fieldnames})

    def log(self, **kwargs):
        if not self.fieldnames:
            self.fieldnames = list(kwargs.keys())
        else:
            new_keys = [key for key in kwargs.keys() if key not in self.fieldnames]
            if new_keys:
                existing_rows = self._read_existing_rows()
                self.fieldnames.extend(new_keys)
                self._rewrite_with_fieldnames(existing_rows)

        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({key: kwargs.get(key, "") for key in self.fieldnames})


if __name__ == '__main__':
    cfg = load_yaml("configs/dataset_sun_rgb_d.yaml")
