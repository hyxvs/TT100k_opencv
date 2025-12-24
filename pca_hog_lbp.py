import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================== 1. 加载特征 ==================
data = np.load("F:/TT100K/features.npz", allow_pickle=True)
X, y = data["X"], data["y"]

print("Feature shape:", X.shape)

# ================== 2. 特征标准化（非常关键） ==================
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# ================== 3. PCA 降维 ==================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:",
      np.sum(pca.explained_variance_ratio_))

# ================== 4. PCA 可视化 ==================
plt.figure(figsize=(12, 10))

labels = np.unique(y)
for label in labels:
    idx = (y == label)
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        s=10,
        alpha=0.7,
        label=label
    )

plt.title("PCA Visualization of HOG + LBP Features", fontsize=13)
plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)"
)
plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)"
)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=18,             # 每行 18 个
    fontsize=8,
    frameon=False,
    columnspacing=1.5
)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

# ================== 5. 保存图片（可选，强烈建议） ==================
plt.savefig("F:/TT100K/pca_hog_lbp.png", dpi=300,bbox_inches="tight")

plt.show()
