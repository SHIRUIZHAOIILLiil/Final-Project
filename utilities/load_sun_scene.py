import yaml, csv, torch, os
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
        self.header_written = os.path.exists(filepath)

    def log(self, **kwargs):
        write_header = not self.header_written
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=kwargs.keys())
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(kwargs)


if __name__ == '__main__':
    cfg = load_yaml("configs/dataset_sun_rgb_d.yaml")