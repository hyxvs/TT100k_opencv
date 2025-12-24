import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ================== 1. 加载特征 ==================
data = np.load("F:/TT100K/features.npz", allow_pickle=True)
X, y = data["X"], data["y"]

# ================== 2. 标准化 ==================
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# ================== 3. t-SNE（全量） ==================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

X_tsne = tsne.fit_transform(X_std)

# ================== 4. 可视化 ==================
plt.figure(figsize=(12, 10))

labels = np.unique(y)
for label in labels:
    mask = y == label
    plt.scatter(
        X_tsne[mask, 0],
        X_tsne[mask, 1],
        s=10,
        alpha=0.7,
        label=label
    )

plt.title("t-SNE Visualization of HOG + LBP Features", fontsize=13)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 图例铺平
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=18,
    fontsize=8,
    frameon=False
)

plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

# 保存
plt.savefig("F:/TT100K/tsne_hog_lbp_full.png", dpi=300, bbox_inches="tight")
plt.show()
