import yaml
from pathlib import Path

def load_yaml(path: str) -> dict:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



if __name__ == '__main__':
    cfg = load_yaml("configs/dataset_sun_rgb_d.yaml")