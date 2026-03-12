# Week02 Practice README

이 README 파일은 week02 폴더의 practice04.py, practice05.py, practice06.py 파일들의 전체 코드, 주요 코드 설명, 그리고 실행 결과를 포함합니다.

# Practice 04: 체크보드 기반 카메라 캘리브레이션

## 전체 코드

```python
# OpenCV 라이브러리 (이미지 처리 및 카메라 캘리브레이션 기능 제공)
import cv2

# 수치 연산 및 배열 처리를 위한 NumPy
import numpy as np

# 특정 패턴에 맞는 파일 목록을 가져오기 위한 라이브러리
import glob

# 파일 경로 및 디렉토리 처리를 위한 라이브러리
import os


# 체크보드 내부 코너 개수 설정 (가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 크기 (mm 단위)
square_size = 25.0

# 코너 위치를 서브픽셀 수준으로 정밀화할 때 사용할 종료 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# 체크보드 코너 개수만큼 3D 좌표 공간 생성 (초기값은 0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)

# 체스보드 격자 좌표 생성 (z=0 평면에 있다고 가정)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 실제 체스보드 한 칸 크기를 반영하여 좌표를 mm 단위로 변환
objp *= square_size


# 실제 세계 좌표(3D 좌표)를 저장할 리스트
objpoints = []

# 이미지에서 검출된 코너 좌표(2D 좌표)를 저장할 리스트
imgpoints = []


# 현재 실행 중인 Python 파일의 절대 경로를 얻고 디렉토리만 추출
base_dir = os.path.dirname(os.path.abspath(__file__))

# calibration_images 폴더 안의 left*.jpg 이미지 경로 패턴 생성
image_pattern = os.path.join(base_dir, "../calibration_images/left*.jpg")

# 패턴에 해당하는 모든 이미지를 찾고 정렬
images = sorted(glob.glob(image_pattern))


# 이미지 크기를 저장할 변수 (캘리브레이션 계산에 필요)
img_size = None


# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------

# 모든 캘리브레이션 이미지를 순서대로 처리
for fname in images:

    # 이미지 파일을 읽어오기
    img = cv2.imread(fname)

    # 코너 검출 정확도를 높이기 위해 grayscale 이미지로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 첫 번째 이미지에서 이미지 크기를 저장
    if img_size is None:
        img_size = gray.shape[::-1]

    # 이미지에서 체크보드 내부 코너 검출
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너 검출에 성공한 경우
    if ret == True:

        # 실제 세계 좌표(3D)를 리스트에 저장
        objpoints.append(objp)
        
        # 코너 위치를 서브픽셀 수준으로 정밀하게 보정
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 보정된 코너 좌표를 이미지 좌표 리스트에 저장
        imgpoints.append(corners2)

        # 검출된 코너를 이미지 위에 표시
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        # 코너 검출 결과를 화면에 출력
        cv2.imshow('Checking Corners', img)
        
        # 100ms 동안 화면에 표시
        cv2.waitKey(100)


# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()


# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------

# 캘리브레이션 수행하여 카메라 내부 파라미터와 왜곡 계수 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)


# 카메라 내부 파라미터 행렬 출력
print("Camera Matrix K:")
print(K)

# 렌즈 왜곡 계수 출력
print("\nDistortion Coefficients:")
print(dist)


# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------

# 캘리브레이션에 사용한 첫 번째 이미지를 불러오기
sample_img = cv2.imread(images[0])

# 계산된 카메라 행렬과 왜곡 계수를 이용하여 이미지 왜곡 보정
dst = cv2.undistort(sample_img, K, dist, None, K)


# 원본 이미지 출력
cv2.imshow('Original Image', sample_img)

# 왜곡 보정된 이미지 출력
cv2.imshow('Undistorted Image', dst)


# 키 입력이 있을 때까지 화면 유지
cv2.waitKey(0)

# 모든 OpenCV 창 종료
cv2.destroyAllWindows()
```

## 주요 코드 설명

### 1. 체스보드 패턴 설정

```python
CHECKERBOARD = (9, 6)
square_size = 25.0
```

카메라 캘리브레이션에서는 **체스보드 패턴 이미지**를 사용하여  
카메라의 내부 파라미터를 계산합니다.

- `CHECKERBOARD = (9,6)`
  - 체스보드 **내부 코너 개수**를 의미합니다.
  - 가로 9개, 세로 6개의 코너를 검출합니다.

주의할 점

- 체스보드 **칸 개수가 아니라 코너 개수**입니다.

예시

```
체스보드 칸: 10 × 7
내부 코너: 9 × 6
```

`square_size`는 체스보드 한 칸의 실제 크기입니다.

이 값은 **실제 거리 계산 스케일을 결정하는 중요한 파라미터**입니다.

---

### 2. 3D World 좌표 생성

```python
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3),np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size
```

캘리브레이션을 위해 **체스보드 코너의 실제 세계 좌표**를 생성합니다.

좌표 구조

```
(X, Y, Z)
```

하지만 체스보드는 평면이기 때문에

```
Z = 0
```

으로 설정됩니다.

예시

```
(0,0,0)
(1,0,0)
(2,0,0)
...
```

이 좌표에 `square_size`를 곱하여 **실제 물리적 크기(mm)** 를 반영합니다.

---

### 3. 체스보드 코너 검출

```python
ret, corners = cv2.findChessboardCorners(gray,CHECKERBOARD,None)
```

OpenCV의 `findChessboardCorners()` 함수는

- 체스보드 패턴을 탐지하고
- 내부 코너 위치를 검출합니다.

반환값

- `ret` : 검출 성공 여부
- `corners` : 검출된 코너 좌표

성공한 이미지에서만 캘리브레이션에 사용됩니다.

---

### 4. 코너 정밀화 (Subpixel Refinement)

```python
corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
```

체스보드 코너 위치를 **서브픽셀 수준으로 보정**합니다.

이 과정이 필요한 이유

- `findChessboardCorners()` 결과는 **픽셀 단위 정확도**
- `cornerSubPix()`는 **서브픽셀 정확도**

즉 **더 정확한 캘리브레이션 결과를 얻을 수 있습니다.**

---

### 5. 카메라 캘리브레이션 수행

```python
ret,K,dist,rvecs,tvecs = cv2.calibrateCamera(
    objpoints,imgpoints,img_size,None,None
)
```

OpenCV의 `calibrateCamera()` 함수는  
체스보드 이미지들을 이용하여 **카메라 파라미터를 계산합니다.**

입력

- `objpoints` : 실제 3D 좌표
- `imgpoints` : 이미지에서 검출된 2D 좌표

출력

- `K` : 카메라 내부 파라미터 행렬
- `dist` : 렌즈 왜곡 계수

카메라 행렬 구조

```
[ fx  0  cx ]
[ 0  fy  cy ]
[ 0   0   1 ]
```

- `fx, fy` : 초점 거리
- `cx, cy` : 이미지 중심

---

### 6. 렌즈 왜곡 보정

```python
dst = cv2.undistort(sample_img,K,dist)
```

카메라 렌즈는 일반적으로 다음과 같은 왜곡이 발생합니다.

- Barrel distortion
- Pincushion distortion

`cv2.undistort()`는 캘리브레이션 결과를 이용하여  
**이미지 왜곡을 제거합니다.**

---

### 실행 결과
<img width="1260" height="1004" alt="image" src="https://github.com/user-attachments/assets/3ce9ad06-b9ee-4327-a923-c2bc00bcc65b" />

<img width="1267" height="996" alt="image" src="https://github.com/user-attachments/assets/c4ea68bd-9311-4506-b8c3-67e06f5f11fe" />

<img width="962" height="249" alt="image" src="https://github.com/user-attachments/assets/8b8b734b-4fa9-4439-b32f-7f9533f3cfa7" />

# Practice 05: 이미지 Rotation & Transformation

## 전체 코드

```python
# OpenCV 라이브러리 (이미지 처리 및 기하 변환 기능 제공)
import cv2

# 수치 계산 및 배열 처리를 위한 NumPy 라이브러리
import numpy as np

# 파일 경로 처리를 위한 OS 라이브러리
import os


# 현재 실행 중인 Python 파일의 절대 경로를 얻고 그 디렉토리를 추출
base_dir = os.path.dirname(os.path.abspath(__file__))

# 현재 디렉토리 기준 상위 폴더에 있는 soccer.jpg 파일 경로 생성
image_path = os.path.join(base_dir, "../rose.png")


# 지정한 경로에서 이미지를 읽어오기
img = cv2.imread(image_path)

# 이미지가 정상적으로 로드되지 않았을 경우 예외 처리
if img is None:
    print(f"Error: Could not read image at {image_path}")
    exit()


# 이미지의 높이(h)와 너비(w) 추출
h, w = img.shape[:2]

# 이미지 중심 좌표 계산 (회전 기준점)
center = (w // 2, h // 2)


# -----------------------------
# 2. 회전 및 크기 조절 (Rotation & Scaling)
# -----------------------------

# 회전 각도 설정 (30도)
angle = 30

# 이미지 크기 스케일 설정 (80% 크기)
scale = 0.8

# 중심 기준으로 회전 + 크기 조절을 위한 2x3 변환 행렬 생성
M = cv2.getRotationMatrix2D(center, angle, scale)


# -----------------------------
# 3. 평행 이동 (Translation)
# -----------------------------

# x 방향 이동 거리 (오른쪽으로 80픽셀 이동)
tx = 80

# y 방향 이동 거리 (위쪽으로 40픽셀 이동)
ty = -40

# 변환 행렬의 x 방향 이동 성분에 tx 추가
M[0, 2] += tx

# 변환 행렬의 y 방향 이동 성분에 ty 추가
M[1, 2] += ty


# -----------------------------
# 4. 아핀 변환 적용 (Affine Transformation)
# -----------------------------

# 변환 행렬 M을 이용해 이미지에 아핀 변환 적용
dst = cv2.warpAffine(img, M, (w, h))


# -----------------------------
# 5. 결과 시각화
# -----------------------------

# 원본 이미지 출력
cv2.imshow('Original Image', img)

# 변환된 이미지 출력
cv2.imshow('Transformed Image', dst)

# 키 입력이 있을 때까지 창 유지
cv2.waitKey(0)

# OpenCV 창 모두 닫기
cv2.destroyAllWindows()
```

## 주요 코드 설명

### 1. 이미지 불러오기

```python
img = cv2.imread(image_path)
```

OpenCV의 `cv2.imread()` 함수를 이용하여 이미지를 읽어옵니다.

- 이미지는 **NumPy 배열 형태**로 저장됩니다.
- OpenCV는 기본적으로 **BGR 색상 순서**를 사용합니다.  
  (일반적인 RGB와 다르기 때문에 주의해야 합니다.)
- 파일 경로가 잘못되었거나 이미지가 존재하지 않으면 `None`을 반환합니다.

따라서 다음과 같이 예외 처리를 통해 **이미지 로드 실패 상황을 방지합니다.**

```python
if img is None:
    exit()
```

---

### 2. 이미지 크기와 중심 좌표 계산

```python
h, w = img.shape[:2]
center = (w // 2, h // 2)
```

이미지의 높이와 너비를 가져와 **중심 좌표를 계산합니다.**

- `img.shape` → `(height, width, channels)`
- `shape[:2]` → 높이와 너비만 추출

이미지 중심 좌표는 **회전 변환의 기준점**으로 사용됩니다.

---

### 3. 회전 및 스케일 변환 행렬 생성

```python
M = cv2.getRotationMatrix2D(center, angle, scale)
```

OpenCV의 `getRotationMatrix2D()` 함수는  
이미지 회전을 위한 **2×3 변환 행렬**을 생성합니다.

매개변수

- `center` : 회전 기준점
- `angle` : 회전 각도 (도 단위)
- `scale` : 확대 또는 축소 비율

예시

```
angle = 30
scale = 0.8
```

→ 이미지를 **30도 회전하고 0.8배 축소**합니다.

---

### 4. 평행 이동 적용

```python
M[0,2] += tx
M[1,2] += ty
```

회전 행렬의 마지막 열은 **평행 이동(translation)** 값을 의미합니다.

```
[ a  b  tx ]
[ c  d  ty ]
```

여기서

- `tx` : x 방향 이동량
- `ty` : y 방향 이동량

예시

```
tx = 80
ty = -40
```

→ 이미지가 **오른쪽 80px, 위쪽 40px 이동**합니다.

---

### 5. 아핀 변환 적용

```python
dst = cv2.warpAffine(img, M, (w, h))
```

OpenCV의 `warpAffine()` 함수는  
**아핀 변환(Affine Transformation)** 을 이미지에 적용합니다.

아핀 변환에는 다음 변환이 포함됩니다.

- 회전 (Rotation)
- 크기 조절 (Scaling)
- 평행 이동 (Translation)
- 기울기 변환 (Shear)

매개변수

- `img` : 원본 이미지
- `M` : 변환 행렬
- `(w, h)` : 출력 이미지 크기

---

### 6. 결과 이미지 출력

```python
cv2.imshow('Original Image', img)
cv2.imshow('Transformed Image', dst)
```

OpenCV 창을 생성하여 이미지를 출력합니다.

- 첫 번째 창 : 원본 이미지
- 두 번째 창 : 변환된 이미지

이렇게 하면 **변환 전후를 쉽게 비교할 수 있습니다.**

---

### 7. 키 입력 대기

```python
cv2.waitKey(0)
```

OpenCV 창이 바로 닫히지 않도록  
**사용자의 키 입력을 기다립니다.**

- `0` → 무한 대기
- 키 입력이 발생하면 다음 코드 실행

---

### 8. OpenCV 창 종료

```python
cv2.destroyAllWindows()
```

프로그램 종료 시 **열려 있는 모든 OpenCV 창을 닫습니다.**

이 과정이 없으면

- 프로그램 종료 후에도
- OpenCV 창이 남아 있을 수 있습니다.

따라서 **이미지 출력 프로그램에서는 필수적으로 사용되는 함수입니다.**

---

### 실행 결과
<img width="2348" height="1585" alt="image" src="https://github.com/user-attachments/assets/864665c6-ab06-440c-9aca-0bd235e12b9a" />

<img width="2360" height="1600" alt="image" src="https://github.com/user-attachments/assets/92e5ee7b-22df-4b0a-a915-a8d16ba4f7c1" />

# Practice 06: Stereo Disparity 기반 Depth 추정

## 전체 코드

```python
# OpenCV 라이브러리 (스테레오 매칭, 이미지 처리, 시각화에 사용)
import cv2

# 수치 연산 및 배열 계산을 위한 NumPy
import numpy as np

# 파일 경로를 객체 형태로 다루기 위한 pathlib
from pathlib import Path


# 결과 이미지를 저장할 outputs 폴더 생성
output_dir = Path("./outputs")

# 폴더가 없으면 생성 (parents=True: 상위 폴더까지 생성)
output_dir.mkdir(parents=True, exist_ok=True)


# OS 관련 경로 처리를 위한 라이브러리
import os


# 현재 실행 중인 Python 파일의 디렉토리 경로 가져오기
base_dir = Path(__file__).parent.resolve()

# 좌측 이미지 불러오기
left_color = cv2.imread(str(base_dir / "../left.png"))

# 우측 이미지 불러오기
right_color = cv2.imread(str(base_dir / "../right.png"))


# 이미지가 정상적으로 로드되지 않았을 경우 예외 처리
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")


# 카메라 초점 거리 (focal length)
f = 700.0

# 두 카메라 사이 거리 (baseline)
B = 0.12


# 거리 측정을 수행할 관심 영역(ROI) 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}


# 스테레오 매칭을 위해 좌/우 이미지를 grayscale로 변환
gray_left = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)

# 오른쪽 이미지 grayscale 변환
gray_right = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)


# -----------------------------
# 1. Disparity 계산
# -----------------------------

# StereoBM 알고리즘 생성 (numDisparities: 탐색 범위, blockSize: 블록 크기)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# 좌/우 이미지의 disparity 계산
disparity_int = stereo.compute(gray_left, gray_right)

# disparity 값을 float32로 변환하고 16으로 나누어 실제 값으로 복원
disparity = disparity_int.astype(np.float32) / 16.0


# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------

# disparity가 0보다 큰 픽셀만 유효한 데이터로 판단
valid_mask = disparity > 0

# depth map 초기화
depth_map = np.zeros_like(disparity)

# 깊이 계산 공식 적용 (Z = fB / d)
depth_map[valid_mask] = (f * B) / disparity[valid_mask]


# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------

# 결과 저장용 딕셔너리
results = {}

# 각 ROI에 대해 반복 수행
for name, (x, y, w, h) in rois.items():

    # ROI 영역의 disparity 추출
    roi_disp = disparity[y:y+h, x:x+w]

    # ROI 영역의 depth 추출
    roi_depth = depth_map[y:y+h, x:x+w]

    # ROI 영역에서 유효한 disparity 위치 추출
    roi_valid = valid_mask[y:y+h, x:x+w]
    
    # 유효한 픽셀이 존재하는 경우 평균 계산
    if np.any(roi_valid):
        avg_disp = np.mean(roi_disp[roi_valid])
        avg_depth = np.mean(roi_depth[roi_valid])

    # 유효 픽셀이 없는 경우 0으로 설정
    else:
        avg_disp = 0
        avg_depth = 0
    
    # 결과 딕셔너리에 저장
    results[name] = {"avg_disp": avg_disp, "avg_depth": avg_depth}


# -----------------------------
# 4. 결과 출력
# -----------------------------

# 표 형태로 결과 출력
print(f"{'ROI':<10} | {'Avg Disparity':<15} | {'Avg Depth':<10}")
print("-" * 40)

# 각 ROI 결과 출력
for name, data in results.items():
    print(f"{name:<10} | {data['avg_disp']:<15.4f} | {data['avg_depth']:<10.4f}")


# depth 값이 가장 작은 객체 (가장 가까운 객체)
closest_roi = min(results.items(), key=lambda item: item[1]['avg_depth'])

# depth 값이 가장 큰 객체 (가장 먼 객체)
farthest_roi = max(results.items(), key=lambda item: item[1]['avg_depth'])


# 가장 가까운 객체 출력
print(f"\n가장 가까운 객체: {closest_roi[0]} (Avg Depth: {closest_roi[1]['avg_depth']:.4f})")

# 가장 먼 객체 출력
print(f"가장 먼 객체: {farthest_roi[0]} (Avg Depth: {farthest_roi[1]['avg_depth']:.4f}")


# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------

# disparity 복사
disp_tmp = disparity.copy()

# disparity가 0 이하인 값은 NaN 처리
disp_tmp[disp_tmp <= 0] = np.nan


# 모든 값이 NaN이면 오류 발생
if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")


# disparity 값의 하위 5% 값
d_min = np.nanpercentile(disp_tmp, 5)

# disparity 값의 상위 95% 값
d_max = np.nanpercentile(disp_tmp, 95)


# 값이 동일할 경우 대비
if d_max <= d_min:
    d_max = d_min + 1e-6


# disparity 값을 0~1 범위로 정규화
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)

# 범위 제한
disp_scaled = np.clip(disp_scaled, 0, 1)


# 시각화용 disparity 이미지 생성
disp_vis = np.zeros_like(disparity, dtype=np.uint8)

# 유효 disparity 위치
valid_disp = ~np.isnan(disp_tmp)

# 0~255 범위로 변환
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)


# 컬러맵 적용 (JET: 가까울수록 빨강)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)


# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------

# depth 시각화용 이미지 생성
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)


# 유효 depth 값이 존재하는 경우
if np.any(valid_mask):

    # 유효 depth 값 추출
    depth_valid = depth_map[valid_mask]

    # depth 하위 5%
    z_min = np.percentile(depth_valid, 5)

    # depth 상위 95%
    z_max = np.percentile(depth_valid, 95)


    # 값이 동일할 경우 대비
    if z_max <= z_min:
        z_max = z_min + 1e-6


    # depth 값을 0~1 범위로 정규화
    depth_scaled = (depth_map - z_min) / (z_max - z_min)

    # 범위 제한
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 클수록 멀기 때문에 색상 반전
    depth_scaled = 1.0 - depth_scaled

    # 0~255로 변환
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)


# 컬러맵 적용
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------

# 왼쪽 이미지 복사
left_vis = left_color.copy()

# 오른쪽 이미지 복사
right_vis = right_color.copy()


# 각 ROI 영역을 이미지에 표시
for name, (x, y, w, h) in rois.items():

    # 왼쪽 이미지에 ROI 사각형 표시
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ROI 이름 표시
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 오른쪽 이미지에 ROI 표시
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ROI 이름 표시
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# -----------------------------
# 8. 저장
# -----------------------------

# disparity 시각화 이미지 저장
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)

# depth 시각화 이미지 저장
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)

# ROI가 표시된 왼쪽 이미지 저장
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)


# -----------------------------
# 9. 출력
# -----------------------------

# ROI가 표시된 원본 이미지 출력
cv2.imshow("Original Left (with ROIs)", left_vis)

# disparity 시각화 출력
cv2.imshow("Disparity Map", disparity_color)

# depth 시각화 출력
cv2.imshow("Depth Map", depth_color)


# 키 입력 대기
cv2.waitKey(0)

# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()
```

---

## 주요 코드 설명

### 1. StereoBM 객체 생성

```python
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
```

StereoBM(Stereo Block Matching)은  
좌우 이미지의 **disparity(시차)** 를 계산하는 알고리즘입니다.

주요 파라미터

- `numDisparities`
  - 탐색할 disparity 범위
  - **16의 배수**여야 합니다.

- `blockSize`
  - 매칭에 사용하는 블록 크기
  - 클수록 노이즈 감소
  - 작을수록 디테일 증가

---

### 2. Disparity 계산

```python
disparity_int = stereo.compute(gray_left, gray_right)
```

좌우 이미지의 픽셀 위치 차이를 계산합니다.

이 값이 바로 **disparity** 입니다.

disparity 의미

```
왼쪽 이미지 픽셀 위치
-
오른쪽 이미지 픽셀 위치
```

특징

- disparity가 클수록 **가까운 물체**
- disparity가 작을수록 **먼 물체**

---

### 3. Disparity 정규화

```python
disparity = disparity_int.astype(np.float32) / 16.0
```

StereoBM은 disparity 값을

```
16배 스케일된 정수
```

형태로 반환합니다.

따라서 실제 disparity 값을 얻기 위해  
**16으로 나누어야 합니다.**

---

### 4. Depth 계산

```python
depth_map[valid_mask] = (f * B) / disparity[valid_mask]
```

Depth는 다음 공식을 이용하여 계산됩니다.

\[
Z = \frac{fB}{d}
\]

여기서

- `Z` : Depth (거리)
- `f` : 카메라 초점 거리
- `B` : 카메라 baseline
- `d` : disparity

특징

- disparity ↑ → depth ↓ → 가까움
- disparity ↓ → depth ↑ → 멀어짐

---

### 5. ROI 영역 평균 거리 계산

```python
roi_disp = disparity[y:y+h, x:x+w]
roi_depth = depth_map[y:y+h, x:x+w]
```

NumPy 슬라이싱을 이용하여  
**ROI 영역의 disparity와 depth 값을 추출**합니다.

이후

```
평균 disparity
평균 depth
```

를 계산하여 **객체 간 거리 비교**를 수행합니다.

---

### 6. Disparity 시각화

```python
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
```

Disparity 값을 컬러맵으로 변환하여  
거리 정보를 **시각적으로 표현**합니다.

색상 의미

```
빨강 → 가까움
파랑 → 멀리 있음
```

---

### 7. Depth Map 시각화

Depth 값도 동일하게 정규화 후 컬러맵을 적용하여  
**거리 정보를 색상으로 표현**합니다.

단, depth는 값이 클수록 멀기 때문에

```
depth_scaled = 1 - depth_scaled
```

과 같이 **색상을 반전**합니다.

---

### 8. ROI 시각화

```python
cv2.rectangle()
cv2.putText()
```

ROI 영역을 이미지에 표시하여  
**어떤 객체의 거리를 계산했는지 시각적으로 확인할 수 있습니다.**

표시 정보

- ROI 위치
- 객체 이름

---

### 9. 결과 저장

```python
cv2.imwrite()
```

다음 이미지를 파일로 저장합니다.

- disparity_map.png
- depth_map.png
- left_roi.png

이를 통해 **분석 결과를 이미지 파일로 확인할 수 있습니다.**

---

### 실행 결과
<img width="886" height="788" alt="image" src="https://github.com/user-attachments/assets/d3242771-c8f7-4986-8b7c-b98615b1fe8e" />

<img width="876" height="786" alt="image" src="https://github.com/user-attachments/assets/e0155513-a1bf-472e-bf62-2bb3fca2f09f" />

<img width="888" height="789" alt="image" src="https://github.com/user-attachments/assets/73824d10-e386-4adc-aadd-00fc95fc9d61" />

<img width="642" height="277" alt="image" src="https://github.com/user-attachments/assets/fc79b8af-c507-4422-811c-a943511f2c0b" />