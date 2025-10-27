import sys
import time
import cv2
import numpy as np
from siftFeatures import (
    extractSiftFeaturesFromBytes,
    matchDescriptors,
    computeSiftSimilarity,
    calcColorSimilarity
)

# np array (Nx4) -> list of cv2.KeyPoint
def convertKpArrayToKeypoints(kpArr: np.ndarray):
    kpList = []
    if kpArr is None or kpArr.size == 0:
        return kpList
    # kpArr rows: [x, y, size, angle]
    for row in kpArr:
        try:
            x = float(row[0])
            y = float(row[1])
            size = float(row[2]) if not np.isnan(row[2]) else 1.0
            angle = float(row[3]) if not np.isnan(row[3]) else -1.0
            # KeyPoint expects positional args: x, y, size, angle
            kp = cv2.KeyPoint(x, y, size, angle)
        except Exception:
            # fallback: create keypoint with minimal args
            try:
                kp = cv2.KeyPoint(float(row[0]), float(row[1]), 1.0)
            except Exception:
                continue
        kpList.append(kp)
    return kpList

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <image1> <image2>")
        sys.exit(1)

    pathA = sys.argv[1]
    pathB = sys.argv[2]

    # load files
    with open(pathA, "rb") as f:
        dataA = f.read()
    with open(pathB, "rb") as f:
        dataB = f.read()

    # extract features (measure time)
    t0 = time.time()
    t = time.time()
    kpArrA, descA, colorA = extractSiftFeaturesFromBytes(dataA)
    extractA = round((time.time() - t) * 1000, 2)

    t = time.time()
    kpArrB, descB, colorB = extractSiftFeaturesFromBytes(dataB)
    extractB = round((time.time() - t) * 1000, 2)

    if descA is None or descB is None or descA.size == 0 or descB.size == 0:
        print("Error: failed to extract descriptors from one of the images.")
        sys.exit(1)

    # matching (measure time)
    t = time.time()
    goodMatches = matchDescriptors(descA, descB)
    matchTime = round((time.time() - t) * 1000, 2)

    # compute similarities
    matchCount = len(goodMatches)
    siftSim = computeSiftSimilarity(matchCount, int(kpArrA.shape[0]), int(kpArrB.shape[0]))
    colorSim = calcColorSimilarity(colorA, colorB)
    totalSim = siftSim * 0.7 + colorSim * 0.3

    totalTime = round((time.time() - t0) * 1000, 2)

    # print summary
    print(f"[Summary]")
    print(f"ImageA: {pathA} kp={int(kpArrA.shape[0])} desc={descA.shape[0]} extract_ms={extractA}")
    print(f"ImageB: {pathB} kp={int(kpArrB.shape[0])} desc={descB.shape[0]} extract_ms={extractB}")
    print(f"MatchCount: {matchCount} match_ms={matchTime}")
    print(f"SIFT similarity: {siftSim:.2f}%")
    print(f"Color similarity: {colorSim:.2f}%")
    print(f"Total similarity: {totalSim:.2f}%")
    print(f"Total time: {totalTime} ms")

    # prepare visualization
    kpListA = convertKpArrayToKeypoints(kpArrA)
    kpListB = convertKpArrayToKeypoints(kpArrB)

    # draw top matches (limit to 200 for visualization)
    matchesToDraw = goodMatches[:200]

    # load color images for drawing
    imgA = cv2.imdecode(np.frombuffer(dataA, np.uint8), cv2.IMREAD_COLOR)
    imgB = cv2.imdecode(np.frombuffer(dataB, np.uint8), cv2.IMREAD_COLOR)

    # draw matches
    drawImg = cv2.drawMatches(imgA, kpListA, imgB, kpListB, matchesToDraw, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # overlay text
    textLines = [
        f"TotalSim: {totalSim:.2f}%",
        f"SIFT: {siftSim:.2f}%  Color: {colorSim:.2f}%",
        f"Matches: {matchCount}  match_ms: {matchTime}ms",
        f"ExtractA_ms: {extractA}  ExtractB_ms: {extractB}",
        f"Total_ms: {totalTime}"
    ]
    x, y = 10, 25
    for line in textLines:
        cv2.putText(drawImg, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y += 25

    # show GUI
    winName = "SIFT Match"
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, drawImg)
    print("Press any key on the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()