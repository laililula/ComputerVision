# OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import cv2 as cv

# 배열 생성과 좌표 데이터 처리를 위해 NumPy를 np라는 이름으로 불러옵니다.
import numpy as np

# 이미지 시각화를 위한 matplotlib의 pyplot을 plt라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt

# 파일 및 폴더 경로 처리를 위한 os 모듈을 불러옵니다.
import os

# 프로그램 종료 등을 위해 sys 모듈을 불러옵니다.
import sys


# 첫 번째 이미지를 읽어서 img1 변수에 저장합니다.
img1 = cv.imread('img1.jpg')

# 두 번째 이미지를 읽어서 img2 변수에 저장합니다.
img2 = cv.imread('img2.jpg')


# 첫 번째 이미지를 불러오지 못했으면 오류 메시지를 출력하고 프로그램을 종료합니다.
if img1 is None:
    print('Failed to load image1.')
    exit()

# 두 번째 이미지를 불러오지 못했으면 오류 메시지를 출력하고 프로그램을 종료합니다.
if img2 is None:
    print('Failed to load image2.')
    exit()


# OpenCV는 기본적으로 BGR 형식을 사용하므로,
# matplotlib에서 올바르게 보이도록 첫 번째 이미지를 RGB 형식으로 변환합니다.
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

# OpenCV는 기본적으로 BGR 형식을 사용하므로,
# matplotlib에서 올바르게 보이도록 두 번째 이미지를 RGB 형식으로 변환합니다.
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)


# SIFT 특징점 검출기를 생성합니다.
sift = cv.SIFT_create()

# 첫 번째 이미지에서 특징점(kp1)을 검출하고,
# 각 특징점에 대한 디스크립터(des1)를 계산합니다.
kp1, des1 = sift.detectAndCompute(img1, None)

# 두 번째 이미지에서 특징점(kp2)을 검출하고,
# 각 특징점에 대한 디스크립터(des2)를 계산합니다.
kp2, des2 = sift.detectAndCompute(img2, None)


# Brute-Force Matcher 객체를 생성합니다.
# NORM_L2는 SIFT 디스크립터에 적합한 거리 계산 방식입니다.
bf = cv.BFMatcher(cv.NORM_L2)

# 각 특징점에 대해 가장 가까운 2개의 매칭 후보를 찾습니다.
matches = bf.knnMatch(des1, des2, k=2)


# Lowe's ratio test를 통과한 좋은 매칭점만 저장할 리스트를 생성합니다.
good_matches = []

# 각 특징점에 대해 가장 가까운 두 후보 m, n을 비교합니다.
for m, n in matches:
    # 첫 번째 후보의 거리가 두 번째 후보보다 충분히 작으면
    # 올바른 매칭일 가능성이 높다고 판단하여 good_matches에 추가합니다.
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)


# 호모그래피를 계산하기 위해 필요한 최소 매칭점 개수를 4개로 설정합니다.
MIN_MATCH_COUNT = 4

# 좋은 매칭점 개수가 최소 개수보다 적으면 호모그래피를 계산할 수 없으므로 프로그램을 종료합니다.
if len(good_matches) < MIN_MATCH_COUNT:
    sys.exit(f"호모그래피를 계산하기 위한 좋은 매칭점이 부족합니다. ({len(good_matches)}개)")


# 첫 번째 이미지에서 좋은 매칭점들의 좌표를 추출하여 src_pts에 저장합니다.
# queryIdx는 첫 번째 이미지의 특징점 인덱스를 의미합니다.
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 두 번째 이미지에서 좋은 매칭점들의 좌표를 추출하여 dst_pts에 저장합니다.
# trainIdx는 두 번째 이미지의 특징점 인덱스를 의미합니다.
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


# RANSAC 방법을 사용하여 두 이미지 사이의 호모그래피 행렬 H를 계산합니다.
# mask는 각 매칭점이 inlier인지 outlier인지 나타내는 정보입니다.
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# 호모그래피 행렬 계산에 실패하면 오류 메시지를 출력하고 프로그램을 종료합니다.
if H is None:
    sys.exit("호모그래피 행렬 계산에 실패했습니다.")


# 첫 번째 이미지의 높이와 너비를 가져옵니다.
h1, w1 = img1.shape[:2]

# 두 번째 이미지의 높이와 너비를 가져옵니다.
h2, w2 = img2.shape[:2]


# 결과 이미지를 위한 전체 가로 크기를 두 이미지 너비의 합으로 설정합니다.
panorama_width = w1 + w2

# 결과 이미지의 세로 크기를 두 이미지 높이 중 더 큰 값으로 설정합니다.
panorama_height = max(h1, h2)


# 첫 번째 이미지를 호모그래피 행렬 H를 이용해 변환하여 warped 이미지로 생성합니다.
warped = cv.warpPerspective(img1_rgb, H, (panorama_width, panorama_height))

# 두 번째 이미지를 warped 결과 이미지의 왼쪽 위 영역에 그대로 배치합니다.
warped[0:h2, 0:w2] = img2_rgb


# mask를 1차원 리스트로 변환하여 inlier 매칭점만 시각화할 수 있도록 준비합니다.
matches_mask = mask.ravel().tolist()


# 두 이미지 사이의 좋은 매칭 결과를 선으로 연결하여 시각화합니다.
# matchesMask를 사용하여 RANSAC에서 inlier로 판정된 매칭점만 그립니다.
matching_result = cv.drawMatches(
    img1_rgb, kp1,
    img2_rgb, kp2,
    good_matches, None,
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# 전체 출력 창의 크기를 가로 18, 세로 8로 설정합니다.
plt.figure(figsize=(18, 8))


# 1행 2열 중 첫 번째 위치에 매칭 결과 이미지를 표시할 영역을 만듭니다.
plt.subplot(1, 2, 1)

# inlier 매칭점이 표시된 결과 이미지를 출력합니다.
plt.imshow(matching_result)

# 첫 번째 이미지의 제목에 inlier 개수와 전체 좋은 매칭 개수를 함께 표시합니다.
plt.title(f"Matching Result (Inliers: {sum(matches_mask)}/{len(good_matches)})")

# 축 눈금과 테두리를 보이지 않도록 설정합니다.
plt.axis("off")


# 1행 2열 중 두 번째 위치에 워핑 결과 이미지를 표시할 영역을 만듭니다.
plt.subplot(1, 2, 2)

# 호모그래피를 이용해 정렬된 결과 이미지를 출력합니다.
plt.imshow(warped)

# 두 번째 이미지의 제목을 설정합니다.
plt.title("Warped Image / Image Alignment")

# 축 눈금과 테두리를 보이지 않도록 설정합니다.
plt.axis("off")


# 그래프 요소들이 겹치지 않도록 여백을 자동 조정합니다.
plt.tight_layout()

# 최종 결과 창을 화면에 표시합니다.
plt.show()