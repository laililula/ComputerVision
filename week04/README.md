# Week04 Practice README

이 README 파일은 week04 폴더의 practice10.py, practice11.py, practice12.py 파일에 대한 내용을 정리한 문서입니다.  
각 실습마다 전체 코드, 주요 코드 설명, 실행 결과 영역을 포함합니다.

---

# Practice 10: SIFT 특징점 검출

## 전체 코드

```python
# OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import cv2 as cv

# 이미지 시각화를 위한 matplotlib의 pyplot을 plt라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt


# 'mot_color70.jpg' 파일을 읽어서 img 변수에 저장합니다.
img = cv.imread('mot_color70.jpg')

# 이미지를 불러오지 못했으면 오류 메시지를 출력하고 프로그램을 종료합니다.
if img is None:
    print('Failed to load image.')
    exit()


# OpenCV는 기본적으로 BGR 형식을 사용하므로, matplotlib에서 올바르게 보이도록 RGB 형식으로 변환합니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)


# SIFT 특징점 검출기를 생성합니다.
# nfeatures=500은 검출할 특징점의 최대 개수를 대략 500개로 제한하는 옵션입니다.
sift = cv.SIFT_create(nfeatures=500)


# 입력 이미지 img에서 특징점(keypoints)을 검출하고,
# 각 특징점에 대한 디스크립터(descriptors)를 계산합니다.
keypoints, descriptors = sift.detectAndCompute(img, None)


# 검출된 특징점을 원본 RGB 이미지 위에 시각화합니다.
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 옵션을 사용하면
# 특징점의 위치뿐 아니라 크기와 방향 정보도 함께 표시됩니다.
img_with_keypoints = cv.drawKeypoints(
    img_rgb, 
    keypoints, 
    None, 
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# 전체 출력 창의 크기를 가로 12, 세로 6으로 설정합니다.
plt.figure(figsize=(12, 6))


# 1행 2열 중 첫 번째 위치에 원본 이미지를 표시할 영역을 만듭니다.
plt.subplot(1, 2, 1)

# 변환된 RGB 원본 이미지를 출력합니다.
plt.imshow(img_rgb)

# 첫 번째 이미지의 제목을 설정합니다.
plt.title('Original Image')

# 축 눈금과 테두리를 보이지 않도록 설정합니다.
plt.axis('off')


# 1행 2열 중 두 번째 위치에 특징점 결과 이미지를 표시할 영역을 만듭니다.
plt.subplot(1, 2, 2)

# 특징점이 표시된 이미지를 출력합니다.
plt.imshow(img_with_keypoints)

# 두 번째 이미지의 제목을 설정합니다.
plt.title('SIFT Keypoints')

# 축 눈금과 테두리를 보이지 않도록 설정합니다.
plt.axis('off')


# 서브플롯 간 간격을 자동으로 조정하여 겹치지 않게 만듭니다.
plt.tight_layout()

# 최종 결과 창을 화면에 표시합니다.
plt.show()
```

## 주요 코드 설명

### 1. SIFT 검출기 생성

```python
sift = cv.SIFT_create(nfeatures=500)
```

SIFT(Scale-Invariant Feature Transform)는 다양한 스케일과 회전에 불변인 특징점을 검출하는 알고리즘입니다.

- `nfeatures=500` : 최대 500개의 특징점을 검출하도록 제한합니다.

### 2. 특징점 검출 및 디스크립터 계산

```python
keypoints, descriptors = sift.detectAndCompute(img, None)
```

입력 이미지에서 SIFT 특징점을 검출합니다.

- `keypoints` : 검출된 특징점의 위치, 크기, 방향 정보  
- `descriptors` : 각 특징점의 특성을 나타내는 128차원 벡터  

### 3. 특징점 시각화

```python
img_with_keypoints = cv.drawKeypoints(
    img_rgb, 
    keypoints, 
    None, 
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
```

검출된 특징점을 이미지에 시각적으로 표시합니다.

- `DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` : 특징점의 위치뿐만 아니라 크기(원의 반지름)와 방향(선의 방향)까지 표시합니다.

### 4. 결과 비교 표시

원본 이미지와 특징점이 표시된 이미지를 나란히 배치해 검출 결과를 직관적으로 비교합니다.

## 실행 결과

<img width="2377" height="1312" alt="image" src="https://github.com/user-attachments/assets/5818af76-ec8e-4d69-9316-48ea5f800aea" />

---

# Practice 11: SIFT 특징점 매칭

## 전체 코드

```python
# OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import cv2 as cv

# 이미지 시각화를 위한 matplotlib의 pyplot을 plt라는 이름으로 불러옵니다.
import matplotlib.pyplot as plt


# 첫 번째 이미지를 읽어서 img1 변수에 저장합니다.
img1 = cv.imread('mot_color70.jpg')

# 두 번째 이미지를 읽어서 img2 변수에 저장합니다.
img2 = cv.imread('mot_color83.jpg')


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
# crossCheck=True는 서로 가장 잘 대응되는 특징점만 매칭에 사용하도록 설정합니다.
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)


# 첫 번째 이미지의 디스크립터와 두 번째 이미지의 디스크립터를 비교하여 특징점을 매칭합니다.
matches = bf.match(des1, des2)


# 매칭된 결과를 distance 값이 작은 순서대로 정렬합니다.
# distance가 작을수록 두 특징점이 더 유사하다고 볼 수 있습니다.
matches = sorted(matches, key=lambda x: x.distance)


# 정렬된 매칭 결과 중 상위 50개만 선택하여 시각화에 사용합니다.
good_matches = matches[:50]


# 두 이미지 사이의 매칭 결과를 선으로 연결하여 시각화합니다.
# NOT_DRAW_SINGLE_POINTS 옵션은 매칭되지 않은 특징점은 그리지 않도록 설정합니다.
matched_img = cv.drawMatches(
    img1_rgb, kp1,
    img2_rgb, kp2,
    good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# 전체 출력 창의 크기를 가로 16, 세로 8로 설정합니다.
plt.figure(figsize=(16, 8))

# 매칭 결과 이미지를 출력합니다.
plt.imshow(matched_img)

# 출력 이미지의 제목에 표시된 매칭 개수를 함께 보여줍니다.
plt.title(f"SIFT Feature Matching (Top {len(good_matches)})")

# 축 눈금과 테두리를 보이지 않도록 설정합니다.
plt.axis("off")

# 그래프 요소들이 겹치지 않도록 여백을 자동 조정합니다.
plt.tight_layout()

# 최종 결과 창을 화면에 표시합니다.
plt.show()
```

## 주요 코드 설명

### 1. 두 이미지에서 SIFT 특징점 검출

```python
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
```

두 이미지에서 각각 SIFT 특징점과 디스크립터를 검출합니다.

- 각 특징점마다 128차원 벡터(디스크립터)가 생성되어 특징점의 특성을 나타냅니다.

### 2. Brute-Force Matcher 생성

```python
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
```

두 디스크립터 간의 거리를 계산하여 매칭을 수행합니다.

- `NORM_L2` : 유클리드 거리를 사용하여 유사도를 측정합니다.  
- `crossCheck=True` : 양방향으로 일치하는 점만 매칭에 포함시킵니다.  

### 3. 특징점 매칭

```python
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```

디스크립터 거리가 가까운 순서대로 특징점을 매칭하고 정렬합니다.

- `distance`가 작을수록 두 특징점이 더 유사합니다.

### 4. 좋은 매칭 선택

```python
good_matches = matches[:50]
```

정렬된 매칭 결과 중 상위 50개만 선택하여 시각화합니다.

- 이를 통해 노이즈를 제거하고 신뢰도 높은 매칭만 표시합니다.

### 5. 매칭 결과 시각화

```python
matched_img = cv.drawMatches(
    img1_rgb, kp1,
    img2_rgb, kp2,
    good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```

두 이미지를 나란히 배치하고 매칭된 점들을 선으로 연결하여 표시합니다.

## 실행 결과

<img width="2861" height="1564" alt="image" src="https://github.com/user-attachments/assets/c3243b06-6fa1-4fd3-9c51-4ad44873a31b" />

---

# Practice 12: 호모그래피 계산 및 이미지 정렬

## 전체 코드

```python
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
```

## 주요 코드 설명

### 1. KNN Matching을 이용한 특징점 매칭

```python
matches = bf.knnMatch(des1, des2, k=2)
```

각 특징점마다 가장 가까운 2개의 후보를 찾습니다.

- `k=2` : 각 점마다 2개의 가장 가까운 매칭 후보를 반환합니다.

### 2. Lowe's Ratio Test 적용

```python
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```

첫 번째 후보와 두 번째 후보의 거리 비율을 비교하여 신뢰도 높은 매칭만 선택합니다.

- 비율이 작을수록 첫 번째 매칭이 두 번째보다 훨씬 가깝다는 뜻이므로 신뢰도가 높습니다.
- 0.7은 일반적으로 좋은 매칭을 얻기 위한 임계값입니다.

### 3. 호모그래피 행렬 계산

```python
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
```

RANSAC 알고리즘을 사용하여 두 이미지 사이의 호모그래피(기하학적 변환) 행렬을 계산합니다.

- `H` : 3×3 호모그래피 변환 행렬  
- `mask` : inlier와 outlier를 구분하는 마스크 (1이면 inlier, 0이면 outlier)  
- `cv.RANSAC` : 이상치(outlier)에 강건한 방법  
- `5.0` : 변환 오차의 임계값(픽셀)  

### 4. 호모그래피를 이용한 이미지 변환

```python
warped = cv.warpPerspective(img1_rgb, H, (panorama_width, panorama_height))
```

첫 번째 이미지에 호모그래피 변환을 적용하여 두 번째 이미지와 정렬시킵니다.

- 변환 후 지정된 캔버스 크기로 조정됩니다.

### 5. RANSAC 결과의 Inlier 시각화

```python
matches_mask = mask.ravel().tolist()
matching_result = cv.drawMatches(
    img1_rgb, kp1,
    img2_rgb, kp2,
    good_matches, None,
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matches_mask,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```

RANSAC에서 inlier로 판정된 매칭점만 녹색 선으로 표시합니다.

- outlier는 표시되지 않아 호모그래피 계산에 사용된 신뢰도 높은 매칭만 시각화합니다.

### 6. 결과 비교 표시

매칭 결과(inlier 개수 표시)와 정렬된 이미지를 나란히 배치하여 변환 전후를 비교합니다.

## 실행 결과

<img width="2864" height="1561" alt="image" src="https://github.com/user-attachments/assets/adc765c4-98e0-45f6-8efd-3ee6db320000" />

