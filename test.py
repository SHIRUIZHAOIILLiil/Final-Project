import yaml
from datasets import SUNRGBDSceneDataset

import json
from collections import Counter
from pathlib import Path


# def load_yaml(path: str) -> dict:
#     with open(path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)
#
# cfg = load_yaml("configs/dataset_sun_rgb_d.yaml")
#
# ds = SUNRGBDSceneDataset(cfg, split="train")
# print("train size:", len(ds))
# print("num classes:", len(ds.label_to_index))
#
# rgb, depth, y, sid = ds[0]
# print("sample:", sid, "label:", int(y))
# print("rgb:", rgb.shape, rgb.min().item(), rgb.max().item())
# print("depth:", depth.shape, depth.min().item(), depth.max().item())
# print("label_to_index (first 10):", list(ds.label_to_index.items()))
#
# print(ds.items[0])

# ds.summarize()

path = Path(r"E:/dataset/SUNRGBD/roi_index/roi_train.jsonl")
labels = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        labels.append(str(s.get("label")))

cnt = Counter(labels)
print("num samples:", len(labels))
print("num unique labels:", len(cnt))
print("top 30 labels:", cnt.most_common(30))

weird = [lab for lab in cnt if any(ch.isdigit() for ch in lab) or "_" in lab or "/" in lab]
print("weird examples:", weird[:50])

print("-----------------------------------------------------------------")

path = Path(r"E:/dataset/SUNRGBD/roi_index/roi_val.jsonl")
labels = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        labels.append(str(s.get("label")))

cnt = Counter(labels)
print("num samples:", len(labels))
print("num unique labels:", len(cnt))
print("top 30 labels:", cnt.most_common(30))


weird = [lab for lab in cnt if any(ch.isdigit() for ch in lab) or "_" in lab or "/" in lab]
print("weird examples:", weird[:50])