import cv2
import numpy as np
import os
import pickle

def extractSiftFeatures(imagePath):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

def buildFeatureDatabase(imageDir, outputPath):
    db = []
    for file in os.listdir(imageDir):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(imageDir, file)
            descriptors = extractSiftFeatures(path)
            if descriptors is not None:
                tag = '-'.join(file.split('-')[:-1])  # 例: cat-blue-001.jpg → cat-blue="cat"
                db.append({"file": file, "tag": tag, "desc": descriptors})
                print(f"Processed: {file} ({len(descriptors)} features)")
    with open(outputPath, 'wb') as f:
        pickle.dump(db, f)
    print(f"Feature database saved: {outputPath}")

if __name__ == "__main__":
    buildFeatureDatabase("./images", "features.pkl")
