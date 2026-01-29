import yaml
from pathlib import Path
import torch

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


if __name__ == '__main__':
    cfg = load_yaml("configs/dataset_sun_rgb_d.yaml")