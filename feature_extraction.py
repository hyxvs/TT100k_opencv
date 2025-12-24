import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

DATA_DIR = r"F:/TT100K/processed"
OUTPUT_FILE = r"F:/TT100K/features.npz"
IMG_SIZE = 64

def extract_features():
    X, y, paths = [], [], []

    labels = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    total = sum(
        len(os.listdir(os.path.join(DATA_DIR, label)))
        for label in labels
    )

    print(f"Total images: {total}")
    count = 0

    for label in labels:
        label_path = os.path.join(DATA_DIR, label)

        for file in os.listdir(label_path):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue

            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # resize + normalize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0

            # HOG
            hog_feat = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )

            # LBP (uniform)
            lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
            n_bins = 10
            lbp_hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, n_bins + 1),
                density=True
            )

            feature = np.hstack([hog_feat, lbp_hist])

            X.append(feature)
            y.append(label)
            paths.append(img_path)

            count += 1
            if count % 50 == 0:
                print(f"Processed {count}/{total}")

    X = np.array(X)
    y = np.array(y)
    paths = np.array(paths)

    np.savez_compressed(OUTPUT_FILE, X=X, y=y, paths=paths)
    print(f"✅ Features saved to {OUTPUT_FILE}")
    print(f"Total processed images: {count}")

    return X, y, paths

if __name__ == "__main__":
    extract_features()
