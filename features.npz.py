import numpy as np

# ================== 加载特征文件 ==================
data = np.load("F:/TT100K/features.npz", allow_pickle=True)
X = data["X"]       # 特征矩阵
y = data["y"]       # 类别标签
paths = data["paths"]  # 原始图像路径

# ================== 查看基本信息 ==================
print("X shape:", X.shape)
print("y shape:", y.shape)
print("paths shape:", paths.shape)
print("类别示例:", np.unique(y)[:10])
