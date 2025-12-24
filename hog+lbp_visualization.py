import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern

# ================== 0. 加载 features.npz ==================
data = np.load("F:/TT100K/features.npz", allow_pickle=True)
paths = data["paths"]

# ================== 1. 读取原始图像 ==================
img_path = paths[0]
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64))

# ================== 2. HOG 计算 ==================
hog_feat, hog_img = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True
)

hog_img = (hog_img - hog_img.min()) / (hog_img.max() - hog_img.min() + 1e-6)

# ================== 3. LBP 计算（uniform） ==================
lbp = local_binary_pattern(
    img,
    P=8,
    R=1,
    method="uniform"
)

# 为了显示效果进行归一化
lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-6)

# ================== 4. 可视化 ==================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(hog_img, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(lbp_norm, cmap="gray")
plt.title("LBP Pattern Image")
plt.axis("off")

plt.suptitle("HOG and LBP Feature Visualization", fontsize=12)
plt.tight_layout()
plt.show()
