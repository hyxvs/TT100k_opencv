# -*- coding: utf-8 -*-
"""
Plot model comparison bar chart from CSV
"""

import matplotlib.pyplot as plt
import csv

CSV_FILE = r"F:/TT100K/model_comparison.csv"
OUTPUT_IMG = r"F:/TT100K/model_comparison.png"

# ================== 1️⃣ 读取 CSV ==================
models = []
accuracies = []

with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Model'] in ['SVM', 'CNN']:
            models.append(row['Model'])
            accuracies.append(float(row['Accuracy']))

# ================== 2️⃣ 绘制柱状图 ==================
plt.figure(figsize=(5,5))
bars = plt.bar(models, accuracies, color=['skyblue', 'salmon'], alpha=0.7,width=0.5)
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Comparison on TT100K')
plt.grid(axis='y', linestyle='--', alpha=0.5, color='gray')

# 在柱上显示准确率
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{acc:.4f}", ha='center', va='bottom', fontsize=12)

# 保存图片
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)
plt.show()

print(f"Bar chart saved to {OUTPUT_IMG}")
