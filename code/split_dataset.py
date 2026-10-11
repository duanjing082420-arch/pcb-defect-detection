"""把 PCB 缺陷检测数据集按 8:2 随机切分到 train/val 子目录."""
import os
import random
import shutil
from pathlib import Path

# --- 配置 (相对脚本位置, 整个文件夹移到哪都能跑) ---
BASE = Path(__file__).resolve().parent.parent      # 项目根目录
DATA_ROOT = BASE / "dataset"
IMG_SRC = DATA_ROOT / "images"
LBL_SRC = DATA_ROOT / "labels"

# 目标: 把 images 和 label 拍平到 train/val
IMG_TRAIN = IMG_SRC / "train"
IMG_VAL = IMG_SRC / "val"
LBL_TRAIN = LBL_SRC / "train"
LBL_VAL = LBL_SRC / "val"

SPLIT_RATIO = 0.8   # 8:2
SEED = 42           # 固定随机种子，保证可复现
IMG_EXT = ".jpg"

random.seed(SEED)

# 收集全部图片名(不含后缀)
all_imgs = sorted([p.stem for p in IMG_SRC.glob(f"*{IMG_EXT}")])
print(f"原始图片数: {len(all_imgs)}")

# 检查标注一一对应
all_lbls = sorted([p.stem for p in LBL_SRC.glob("*.txt") if p.stem != "classes"])
missing_lbl = [s for s in all_imgs if s not in all_lbls]
if missing_lbl:
    print(f"⚠️ 以下图片没有对应标注: {missing_lbl[:5]}...")
else:
    print(f"✅ 全部 {len(all_imgs)} 张图片都有标注")

# 创建目标目录
for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    d.mkdir(parents=True, exist_ok=True)

# 随机打乱并切分
random.shuffle(all_imgs)
split_idx = int(len(all_imgs) * SPLIT_RATIO)
train_set = all_imgs[:split_idx]
val_set = all_imgs[split_idx:]
print(f"train: {len(train_set)} 张, val: {len(val_set)} 张")

# 拷贝文件 (这里用 copy 而不是 move, 安全)
def copy_pair(stem, img_dst, lbl_dst):
    src_img = IMG_SRC / f"{stem}{IMG_EXT}"
    src_lbl = LBL_SRC / f"{stem}.txt"
    shutil.copy2(src_img, img_dst / src_img.name)
    shutil.copy2(src_lbl, lbl_dst / src_lbl.name)

for s in train_set:
    copy_pair(s, IMG_TRAIN, LBL_TRAIN)
for s in val_set:
    copy_pair(s, IMG_VAL, LBL_VAL)

print(f"✅ 完成拷贝: train={len(list(IMG_TRAIN.iterdir()))}, val={len(list(IMG_VAL.iterdir()))}")
print(f"   标注: train={len(list(LBL_TRAIN.iterdir()))}, val={len(list(LBL_VAL.iterdir()))}")