import cv2
import numpy as np
import pickle
from typing import Tuple, Optional, List

# SIFTでkeypointsとdescriptorsを抽出（カラーも読み込んで平均色を返す）
def extractSiftFeaturesFromBytes(fileBytes: bytes) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    # バイト→カラー画像
    npImg = np.frombuffer(fileBytes, np.uint8)
    imgColor = cv2.imdecode(npImg, cv2.IMREAD_COLOR)
    if imgColor is None:
        return np.empty((0,4), dtype=np.float32), None, np.array([0,0,0], dtype=np.float32)

    # グレースケールでSIFT抽出
    imgGray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(imgGray, None)

    # keypoints を Nx4 の配列 (x, y, size, angle) にする
    if keypoints:
        kpArr = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints], dtype=np.float32)
    else:
        kpArr = np.empty((0,4), dtype=np.float32)

    # 平均色 (B,G,R)
    colorMean = np.mean(imgColor, axis=(0,1)).astype(np.float32)

    return kpArr, descriptors, colorMean

# descriptors 同士をマッチして良いマッチのリストを返す
def matchDescriptors(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75):
    if desc1 is None or desc2 is None:
        return []
    bf = cv2.BFMatcher()  # L2 for SIFT
    rawMatches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m_n in rawMatches:
        # knnMatch が長さ2を返すとは限らないのでチェック
        if len(m_n) < 2:
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

# SIFT類似度を計算（良いマッチ数 ÷ 平均keypoint数 * 100）
def computeSiftSimilarity(matchCount: int, kpCount1: int, kpCount2: int) -> float:
    if kpCount1 == 0 or kpCount2 == 0:
        return 0.0
    avg = (kpCount1 + kpCount2) / 2.0
    return (matchCount / avg) * 100.0

# 平均色から簡易カラー類似度(%)を計算（距離を正規化して1-dist）
def calcColorSimilarity(colorMean1: np.ndarray, colorMean2: np.ndarray) -> float:
    # colorMean: [B, G, R] in 0..255
    maxDist = np.sqrt(255.0**2 * 3)
    dist = np.linalg.norm(colorMean1 - colorMean2)
    sim = max(0.0, 1.0 - (dist / maxDist))
    return float(sim * 100.0)

# DBを保存（pickle）
def saveFeatureDatabase(featureList: List[dict], path: str):
    with open(path, "wb") as f:
        pickle.dump(featureList, f)

# DBを読み込み（pickle）
def loadFeatureDatabase(path: str) -> List[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)