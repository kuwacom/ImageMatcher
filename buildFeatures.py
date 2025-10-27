import os
import sys
import numpy as np
from siftFeatures import extractSiftFeaturesFromBytes, saveFeatureDatabase

# 画像フォルダをスキャンして features.pkl を作る
def buildFeatureDatabaseFromFolder(folderPath: str, outputPath: str = "features.pkl"):
    featureList = []
    for name in sorted(os.listdir(folderPath)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        filePath = os.path.join(folderPath, name)
        with open(filePath, "rb") as f:
            data = f.read()

        kpArr, descriptors, colorMean = extractSiftFeaturesFromBytes(data)
        if descriptors is None or descriptors.size == 0:
            print(f"[WARN] No descriptors: {name}")
            continue

        # tag はファイル名の先頭部分（例: cat_001.jpg -> cat）
        tag = name.split("_")[0] if "_" in name else os.path.splitext(name)[0]

        entry = {
            "file": name,
            "tag": tag,
            "keypoints": kpArr,         # Nx4 array
            "descriptors": descriptors, # Nx128 array
            "colorMean": colorMean      # 3-vector
        }
        featureList.append(entry)
        print(f"[OK] {name} kp={kpArr.shape[0]} desc={descriptors.shape[0]}")

    saveFeatureDatabase(featureList, outputPath)
    print(f"[SAVED] {outputPath} ({len(featureList)} items)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python buildFeatures.py <image-folder> [output.pkl]")
        sys.exit(1)
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "features.pkl"
    buildFeatureDatabaseFromFolder(folder, out)