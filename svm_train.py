# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ================== 配置 ==================
FEATURE_FILE = r"F:/TT100K/features.npz"          # HOG+LBP 特征文件
SVM_MODEL_PATH = r"F:/TT100K/svm_model_final.pkl"  # 保存训练好的模型路径
MIN_SAMPLES = 2                                     # 保留类别的最小样本数
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ================== 加载特征 ==================
data = np.load(FEATURE_FILE)
X, y = data['X'], data['y']

# ================== 删除样本少于 MIN_SAMPLES 的类别 ==================
counts = Counter(y)
valid_labels = [label for label, c in counts.items() if c >= MIN_SAMPLES]
mask = np.isin(y, valid_labels)
X = X[mask]
y = y[mask]

print(f"Total samples after filtering: {len(y)}")
print(f"Remaining classes: {len(valid_labels)} -> {valid_labels}")

# ================== 标签编码 ==================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ================== 数据标准化 ==================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================== 划分训练集和测试集 ==================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# ================== SVM 模型训练 ==================
svm = SVC(kernel='linear', C=1.0, probability=True)
print("Training SVM...")
svm.fit(X_train, y_train)

# ================== 模型评估 ==================
y_pred = svm.predict(X_test)

# 获取测试集中实际出现的类别
labels_in_test = np.unique(y_test)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=labels_in_test,
    target_names=le.inverse_transform(labels_in_test)
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels_in_test))

# ================== 模型保存 ==================
joblib.dump(svm, SVM_MODEL_PATH)
print(f"\n✅ SVM model saved to {SVM_MODEL_PATH}")

# ================== 总结准确率 ==================
accuracy = np.mean(y_pred == y_test)
print(f"\nOverall test accuracy: {accuracy:.4f}")
