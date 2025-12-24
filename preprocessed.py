# -*- coding: utf-8 -*-
"""
TT100K Traffic Sign Dataset Preprocessing Script
- Read annotations.json
- Crop traffic sign bounding boxes
- Resize to 32x32
- Gaussian blur + grayscale + histogram equalization
- Save by category
"""

import os
import cv2
import json
from tqdm import tqdm

# ================== 路径配置（按你自己的目录改） ==================
ROOT_DIR = r"F:/TT100K_data/data"
ANNOT_FILE = os.path.join(ROOT_DIR, "annotations.json")
OUTPUT_DIR = r"F:/TT100K/processed"

IMG_SIZE = 64
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== 图像预处理函数 ==================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

# ================== 读取标注文件 ==================
with open(ANNOT_FILE, "r", encoding="utf-8") as f:
    annotations = json.load(f)

imgs = list(annotations["imgs"].items())

valid_count = 0
skipped_count = 0

print("Start preprocessing TT100K dataset")
print("Total images:", len(imgs))

# ================== 主循环 ==================
for img_id, img_info in tqdm(imgs, desc="Processing"):

    img_path = os.path.join(ROOT_DIR, img_info["path"])
    if not os.path.exists(img_path):
        skipped_count += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        skipped_count += 1
        continue

    height, width = img.shape[:2]

    for obj in img_info.get("objects", []):
        label = obj["category"]
        bbox = obj["bbox"]

        xmin = max(0, int(bbox["xmin"]))
        ymin = max(0, int(bbox["ymin"]))
        xmax = min(width, int(bbox["xmax"]))
        ymax = min(height, int(bbox["ymax"]))

        if xmax <= xmin or ymax <= ymin:
            continue

        crop = img[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        crop = preprocess(crop)

        label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        save_path = os.path.join(label_dir, str(valid_count) + ".png")
        cv2.imwrite(save_path, crop)

        valid_count += 1

# ================== 结束信息（无 emoji，GBK 安全） ==================
print("====================================")
print("Preprocessing finished")
print("Valid samples:", valid_count)
print("Skipped samples:", skipped_count)
print("Output directory:", OUTPUT_DIR)
print("====================================")
