# Week02 Practice README

이 README 파일은 week02 폴더의 practice04.py, practice05.py, practice06.py 파일들의 전체 코드, 주요 코드 설명, 그리고 실행 결과를 포함합니다.

---

# Practice 04: 체크보드 기반 카메라 캘리브레이션

## 전체 코드

```python
import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (9,6)
square_size = 25.0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3),np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size

objpoints = []
imgpoints = []

base_dir = os.path.dirname(os.path.abspath(__file__))
image_pattern = os.path.join(base_dir,"../calibration_images/left*.jpg")

images = sorted(glob.glob(image_pattern))

img_size = None

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if img_size is None:
        img_size = gray.shape[::-1]

    ret,corners = cv2.findChessboardCorners(gray,CHECKERBOARD,None)

    if ret:

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img,CHECKERBOARD,corners2,ret)
        cv2.imshow("Corners",img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

ret,K,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)

print("Camera Matrix K:")
print(K)

print("Distortion Coefficients:")
print(dist)

sample_img = cv2.imread(images[0])
dst = cv2.undistort(sample_img,K,dist)

cv2.imshow("Original",sample_img)
cv2.imshow("Undistorted",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 주요 코드 설명

### 1. 체스보드 코너 검출

```python
cv2.findChessboardCorners()
```

체스보드 패턴을 이용하여 **이미지에서 코너 좌표를 검출**합니다.

---

### 2. 서브픽셀 정밀화

```python
cv2.cornerSubPix()
```

검출된 코너 좌표를 **더 정밀한 위치로 보정**합니다.

---

### 3. 카메라 캘리브레이션

```python
cv2.calibrateCamera()
```

다음 파라미터를 계산합니다.

- 카메라 내부 파라미터 (Camera Matrix)
- 렌즈 왜곡 계수

---

### 4. 왜곡 보정

```python
cv2.undistort()
```

캘리브레이션 결과를 이용하여 **렌즈 왜곡을 제거**합니다.

---

# Practice 05: 이미지 Rotation & Transformation

## 전체 코드

```python
import cv2
import numpy as np
import os

# 현재 실행 중인 파일의 디렉토리를 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "../soccer.jpg")

# 1. 이미지 로드
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()

# 원본 이미지 크기
h, w = img.shape[:2]
center = (w // 2, h // 2)

# -----------------------------
# 2. 회전 및 크기 조절 (Rotation & Scaling)
# -----------------------------
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D(center, angle, scale)

# -----------------------------
# 3. 평행 이동 (Translation)
# -----------------------------
tx = 80
ty = -40

M[0, 2] += tx
M[1, 2] += ty

# -----------------------------
# 4. 아핀 변환 적용
# -----------------------------
dst = cv2.warpAffine(img, M, (w, h))

# -----------------------------
# 5. 결과 시각화
# -----------------------------
cv2.imshow('Original Image', img)
cv2.imshow('Transformed Image', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 주요 코드 설명

### 1. 이미지 불러오기

```python
img = cv2.imread(image_path)
```

OpenCV의 `cv2.imread()` 함수를 사용하여 이미지를 읽어옵니다.

- 이미지는 **NumPy 배열 형태**로 저장됩니다.
- OpenCV는 기본적으로 **BGR 색상 순서**를 사용합니다.
- 파일이 존재하지 않으면 `None`을 반환합니다.

---

### 2. 회전 변환

```python
M = cv2.getRotationMatrix2D(center, angle, scale)
```

이미지를

- 중심 기준 **30도 회전**
- **0.8배 크기로 축소**

하는 변환 행렬을 생성합니다.

---

### 3. 평행 이동

```python
M[0,2] += tx
M[1,2] += ty
```

이미지를

- x 방향 **+80px**
- y 방향 **−40px**

만큼 이동합니다.

---

### 4. 아핀 변환 적용

```python
dst = cv2.warpAffine(img, M, (w, h))
```

회전, 스케일, 평행 이동이 결합된 **아핀 변환**을 이미지에 적용합니다.

---

# Practice 06: Stereo Disparity 기반 Depth 추정

## 전체 코드

```python
stereo = cv2.StereoBM_create(numDisparities=64,blockSize=15)

disparity_int = stereo.compute(gray_left,gray_right)
disparity = disparity_int.astype(np.float32)/16.0

valid_mask = disparity>0

depth_map = np.zeros_like(disparity)
depth_map[valid_mask] = (f*B)/disparity[valid_mask]
```

---

## 주요 코드 설명

### 1. Disparity 계산

```python
stereo.compute()
```

좌우 이미지의 **픽셀 이동량(disparity)** 을 계산합니다.

- disparity가 클수록 **물체가 가까움**
- disparity가 작을수록 **물체가 멀리 있음**

---

### 2. Depth 계산

Depth는 다음 공식을 사용하여 계산합니다.

\[
Z = \frac{fB}{d}
\]

- **f** : 초점 거리
- **B** : 두 카메라 사이 거리
- **d** : disparity

---

### 3. ROI 거리 계산

각 ROI 영역에 대해

- 평균 disparity
- 평균 depth

를 계산하여 **객체 간 거리 비교**를 수행합니다.

---

### 4. Disparity 시각화

```python
cv2.applyColorMap()
```

Disparity 값을 컬러맵으로 변환합니다.

색상 의미

- **빨강 → 가까움**
- **파랑 → 멀리 있음**

---

### 5. Depth Map 생성

Depth 값을 정규화하여 컬러맵을 적용하면 **거리 정보를 시각적으로 확인**할 수 있습니다.

---