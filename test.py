import yaml
from datasets import SUNRGBDSceneDataset

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_yaml("configs/dataset_sun_rgb_d.yaml")

ds = SUNRGBDSceneDataset(cfg, split="train")
print("train size:", len(ds))
print("num classes:", len(ds.label_to_index))

rgb, depth, y, sid = ds[0]
print("sample:", sid, "label:", int(y))
print("rgb:", rgb.shape, rgb.min().item(), rgb.max().item())
print("depth:", depth.shape, depth.min().item(), depth.max().item())
print("label_to_index (first 10):", list(ds.label_to_index.items()))
