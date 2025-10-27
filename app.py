from fastapi import FastAPI, UploadFile, File
import sys
import uvicorn
import time
from siftFeatures import (
    extractSiftFeaturesFromBytes,
    matchDescriptors,
    computeSiftSimilarity,
    calcColorSimilarity,
    loadFeatureDatabase
)

app = FastAPI(title="SIFT Image Search API")

# DBをロード
FEATURE_DB_PATH = "features.pkl"
featureDB = loadFeatureDatabase(FEATURE_DB_PATH)

# /search: ファイル受け取り -> 類似上位5件を返す
@app.post("/search")
async def searchSimilarImage(file: UploadFile = File(...)):
    timings = {}
    t0 = time.time()

    # 特徴抽出
    t = time.time()
    fileBytes = await file.read()
    queryKp, queryDesc, queryColor = extractSiftFeaturesFromBytes(fileBytes)
    timings["featureExtraction_ms"] = round((time.time() - t) * 1000, 2)

    if queryDesc is None or queryDesc.size == 0:
        return {"error": "No features found in query image"}

    results = []
    tMatchAll = time.time()

    # DBと照合
    for entry in featureDB:
        tMatch = time.time()
        dbDesc = entry["descriptors"]
        dbKp = entry["keypoints"]
        dbColor = entry["colorMean"]

        # マッチング
        good = matchDescriptors(queryDesc, dbDesc)
        matchCount = len(good)

        # SIFT類似度
        siftSim = computeSiftSimilarity(matchCount, int(queryKp.shape[0]), int(dbKp.shape[0]))

        # カラー類似度（平均色ベース）
        colorSim = calcColorSimilarity(queryColor, dbColor)

        # 総合類似度（重み付き）
        totalSim = siftSim * 0.7 + colorSim * 0.3

        results.append({
            "file": entry["file"],              # DBファイル名
            "tag": entry["tag"],                # タグ
            "matchCount": matchCount,           # 良いマッチ数
            "siftSimilarity": round(siftSim, 2),# SIFT類似度 %
            "colorSimilarity": round(colorSim, 2),# カラー類似度 %
            "totalSimilarity": round(totalSim, 2),# 総合類似度 %
            "matchingTime_ms": round((time.time() - tMatch) * 1000, 2) # 個別時間
        })

    timings["matchingTotal_ms"] = round((time.time() - tMatchAll) * 1000, 2)

    # ソートと上位5件抽出
    t = time.time()
    results.sort(key=lambda x: x["totalSimilarity"], reverse=True)
    topResults = results[:5]
    timings["sorting_ms"] = round((time.time() - t) * 1000, 2)

    timings["total_ms"] = round((time.time() - t0) * 1000, 2)

    return {"results": topResults, "timings_ms": timings}

# uv run で動くようにする
if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)