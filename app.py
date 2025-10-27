from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import pickle
import sys
import uvicorn
import time

app = FastAPI(title="SIFT Image Search API")

# 事前に生成した特徴量データをロード
# featureDB は list of dict
with open("features.pkl", "rb") as f:
    featureDB = pickle.load(f)

# 画像バイト列からSIFT特徴量を抽出する関数
def extractSiftFeaturesFromBytes(fileBytes):
    # バイト列をNumPy配列に変換
    npImg = np.frombuffer(fileBytes, np.uint8)
    # デコードしてグレースケールに
    image = cv2.imdecode(npImg, cv2.IMREAD_GRAYSCALE)
    # SIFT特徴量を作成
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# 2つのSIFTディスクリプタをマッチングし、良いマッチ数を返す
def matchDescriptors(desc1, desc2):
    bf = cv2.BFMatcher()
    # 各特徴量に対して最良2つのマッチを取得
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    # Loweのratioテストで良いマッチだけを抽出
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return len(good)

# APIエンドポイント: 類似画像検索
# 類似上位5件
@app.post("/search")
async def searchSimilarImage(file: UploadFile = File(...)):
    timings = {}  # 各処理時間を保存
    startTotal = time.time()

    # 特徴抽出
    start = time.time()
    fileBytes = await file.read()
    queryDesc = extractSiftFeaturesFromBytes(fileBytes)
    timings["featureExtraction"] = round((time.time() - start) * 1000, 2)

    if queryDesc is None:
        return {"error": "No features found in query image"}

    results = []
    # DB画像と照合
    startMatchingAll = time.time()
    for entry in featureDB:
        startMatching = time.time()
        dbDesc = entry["desc"]
        goodMatches = matchDescriptors(queryDesc, dbDesc)
        similarity = goodMatches / min(len(queryDesc), len(dbDesc)) * 100
        matchTime = round((time.time() - startMatching) * 1000, 2)  # ms
        results.append({
            "file": entry["file"],
            "tag": entry["tag"],
            "matchCount": goodMatches,
            "similarity": round(similarity, 2),
            "matchingTime_ms": matchTime  # 個別マッチ時間
        })
    timings["matchingTotal"] = round((time.time() - startMatchingAll) * 1000, 2)

    # 結果を類似度順にソート
    start = time.time()
    results.sort(key=lambda x: x["similarity"], reverse=True)
    topResults = results[:5]
    timings["sorting"] = round((time.time() - start) * 1000, 2)

    timings["total"] = round((time.time() - startTotal) * 1000, 2)
    
    return {
        "results": topResults,
        "timings_ms": timings
    }

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
