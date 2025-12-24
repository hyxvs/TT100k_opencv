# -*- coding: utf-8 -*-
"""
Evaluation script: Compare overall accuracy of SVM (HOG+LBP) vs CNN (ResNet18) on TT100K
and save results to CSV
"""

import os
import csv
import numpy as np
from collections import Counter
import joblib
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ================== 配置 ==================
FEATURE_FILE = r"F:/TT100K/features.npz"
SVM_MODEL_PATH = r"F:/TT100K/svm_model_final.pkl"
CNN_MODEL_PATH = r"F:/TT100K/cnn_model.pth"
DATA_DIR = r"F:/TT100K/processed"
RESULT_CSV = r"F:/TT100K/model_comparison.csv"
MIN_SAMPLES = 2
IMG_SIZE = 64
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 1️⃣ SVM 总体准确率 ==================
print("Loading SVM model and features...")
svm = joblib.load(SVM_MODEL_PATH)
data = np.load(FEATURE_FILE)
X, y = data['X'], data['y']

# 过滤少样本类别
counts = Counter(y)
valid_labels = [label for label, c in counts.items() if c >= MIN_SAMPLES]
mask = np.isin(y, valid_labels)
X = X[mask]
y = y[mask]

# 标签编码和标准化
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分测试集
_, X_test, _, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# SVM 预测
y_pred_svm = svm.predict(X_test)
svm_acc = np.mean(y_pred_svm == y_test)
print(f"SVM Overall Accuracy: {svm_acc:.4f}")

# ================== 2️⃣ CNN 总体准确率 ==================
print("\nLoading CNN model and test dataset...")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# 过滤少样本类别
counts_cnn = Counter([dataset.classes[label] for _, label in dataset.samples])
valid_labels_cnn = [label for label, c in counts_cnn.items() if c >= MIN_SAMPLES]
indices = [i for i, (_, label) in enumerate(dataset.samples) if dataset.classes[label] in valid_labels_cnn]
dataset.samples = [dataset.samples[i] for i in indices]

# 划分测试集
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化 CNN 模型
num_classes = len(dataset.classes)
cnn_model = models.resnet18(pretrained=False)
cnn_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, num_classes)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
cnn_model = cnn_model.to(DEVICE)
cnn_model.eval()

# CNN 预测
y_true_cnn, y_pred_cnn = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = cnn_model(imgs)
        _, preds = outputs.max(1)
        y_true_cnn.extend(labels.cpu().numpy())
        y_pred_cnn.extend(preds.cpu().numpy())

cnn_acc = np.mean(np.array(y_pred_cnn) == np.array(y_true_cnn))
print(f"CNN Overall Accuracy: {cnn_acc:.4f}")

# ================== 3️⃣ 模型对比 ==================
print("\n====================================")
print("Model Comparison Summary:")
print(f"SVM Overall Accuracy: {svm_acc:.4f}")
print(f"CNN Overall Accuracy: {cnn_acc:.4f}")
if cnn_acc > svm_acc:
    conclusion = "CNN performs better on TT100K dataset."
else:
    conclusion = "SVM performs better or similar."
print(f"Conclusion: {conclusion}")
print("====================================")

# ================== 4️⃣ 保存结果到 CSV ==================
with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Accuracy'])
    writer.writerow(['SVM', f"{svm_acc:.4f}"])
    writer.writerow(['CNN', f"{cnn_acc:.4f}"])
    writer.writerow(['Conclusion', conclusion])

print(f"Results saved to {RESULT_CSV}")
