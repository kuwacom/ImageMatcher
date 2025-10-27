import cv2
import sys
import numpy as np

def loadImage(path: str):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Failed to load image from {path}")
        sys.exit(1)
    return image

def extractSiftFeatures(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def matchDescriptors(desc1, desc2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatches.append(m)
    return goodMatches

def compareImages(imgPath1: str, imgPath2: str):
    img1 = loadImage(imgPath1)
    img2 = loadImage(imgPath2)

    kp1, desc1 = extractSiftFeatures(img1)
    kp2, desc2 = extractSiftFeatures(img2)

    if desc1 is None or desc2 is None:
        print("Error: Failed to extract descriptors.")
        return

    goodMatches = matchDescriptors(desc1, desc2)

    similarity = len(goodMatches) / min(len(kp1), len(kp2)) * 100
    print(f"Matched Features: {len(goodMatches)}")
    print(f"Similarity: {similarity:.2f}%")

    if similarity > 10:
        print("Result: Images are similar.")
    else:
        print("Result: Images are different.")

    # Optionally visualize matches
    matchedImage = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, None, flags=2)
    cv2.imshow("SIFT Matches", matchedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 3:
        print("Usage: python sift_compare.py <image1> <image2>")
        sys.exit(1)

    imgPath1 = sys.argv[1]
    imgPath2 = sys.argv[2]
    compareImages(imgPath1, imgPath2)

if __name__ == "__main__":
    main()
