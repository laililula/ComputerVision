# Week03 Practice README

이 README 파일은 week03 폴더의 practice07.py, practice08.py, practice09.py 파일에 대한 내용을 정리한 문서입니다.  
각 실습마다 전체 코드, 주요 코드 설명, 실행 결과 영역을 포함합니다.

---

# Practice 07: Sobel 기반 Edge Magnitude 계산

## 전체 코드

```python
# OpenCV 기능을 사용하기 위해 cv라는 이름으로 불러옵니다.
import cv2 as cv

# 배열 계산과 수치 처리를 위해 NumPy를 불러옵니다.
import numpy as np

# 이미지 결과를 화면에 배치해서 보여주기 위해 pyplot을 사용합니다.
import matplotlib.pyplot as plt

# 현재 파일 위치를 기준으로 이미지 경로를 만들기 위해 os를 사용합니다.
import os


# 실행 중인 파이썬 파일의 위치를 절대경로 형태로 구합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))

# images 폴더 안에 있는 edgeDetectionImage.jpg 파일의 전체 경로를 만듭니다.
image_path = os.path.join(base_dir, "images/edgeDetectionImage.jpg")


# 위에서 만든 경로를 이용해 이미지를 읽어옵니다.
img = cv.imread(image_path)

# 이미지가 제대로 읽히지 않았다면
if img is None:
    # 어떤 경로에서 실패했는지 함께 출력합니다.
    print(f"Error: Could not read image at {image_path}")
    # 더 이상 진행하지 않고 프로그램을 종료합니다.
    exit()


# OpenCV는 BGR 순서를 사용하므로, matplotlib에서 올바르게 보이도록 RGB로 바꿉니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)


# 에지 검출을 쉽게 하기 위해 컬러 이미지를 흑백 영상으로 변환합니다.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# x방향 변화량을 계산해서 세로 방향 경계가 얼마나 강한지 구합니다.
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)

# y방향 변화량을 계산해서 가로 방향 경계가 얼마나 강한지 구합니다.
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)


# x방향 기울기와 y방향 기울기를 합쳐 전체 에지 세기를 계산합니다.
magnitude = cv.magnitude(sobel_x, sobel_y)


# 계산된 에지 세기 값을 화면에 표시하기 쉽도록 8비트 영상으로 바꿉니다.
magnitude = cv.convertScaleAbs(magnitude)


# 결과를 한 화면에 보기 위해 figure 크기를 설정합니다.
plt.figure(figsize=(12, 6))


# 1행 2열 중 첫 번째 영역에 원본 이미지를 배치합니다.
plt.subplot(1, 2, 1)

# 원본 이미지를 화면에 출력합니다.
plt.imshow(img_rgb)

# 첫 번째 그림 제목을 설정합니다.
plt.title('Original Image')

# 좌표축 표시를 없애서 이미지만 보이게 합니다.
plt.axis('off')


# 1행 2열 중 두 번째 영역에 에지 강도 이미지를 배치합니다.
plt.subplot(1, 2, 2)

# 계산한 에지 강도 영상을 회색조 형태로 출력합니다.
plt.imshow(magnitude, cmap='gray')

# 두 번째 그림 제목을 설정합니다.
plt.title('Edge Magnitude')

# 이 그림도 축은 숨깁니다.
plt.axis('off')


# 두 이미지의 간격이 겹치지 않도록 자동으로 정리합니다.
plt.tight_layout()

# 현재 결과 화면을 PNG 파일로 저장합니다.
plt.savefig('과제1_결과.png', dpi=300, bbox_inches='tight')

# figure를 닫아 메모리를 정리합니다.
plt.close()
```

## 주요 코드 설명

### 1. 그레이스케일 변환

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

에지 검출 전에 컬러 이미지를 흑백 영상으로 변환합니다.  
이 과정을 거치면 계산량을 줄일 수 있고, 밝기 변화만으로 경계를 찾기 쉬워집니다.

### 2. Sobel 필터 적용

```python
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
```

Sobel 연산은 이미지의 밝기 변화량을 구해 에지를 검출하는 방법입니다.

- `sobel_x` : x방향 미분 → 세로 경계 검출  
- `sobel_y` : y방향 미분 → 가로 경계 검출  

### 3. 에지 강도 계산

```python
magnitude = cv.magnitude(sobel_x, sobel_y)
```

x방향 기울기와 y방향 기울기를 합쳐서 전체 에지 세기를 계산합니다.  
값이 클수록 해당 위치의 경계가 더 뚜렷하다는 뜻입니다.

### 4. 결과 영상 변환

```python
magnitude = cv.convertScaleAbs(magnitude)
```

Sobel 결과는 실수형이므로 바로 화면에 출력하기 어렵습니다.  
그래서 8비트 영상으로 변환해 시각화합니다.

### 5. 결과 시각화

```python
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)

plt.subplot(1, 2, 2)
plt.imshow(magnitude, cmap='gray')
```

원본 이미지와 에지 강도 영상을 나란히 배치하여 비교합니다.

## 실행 결과
<img width="2143" height="661" alt="image" src="https://github.com/user-attachments/assets/c7eb6db2-43b5-4906-9ba0-3524155d8d5f" />

---

# Practice 08: Canny와 Hough Transform을 이용한 직선 검출

## 전체 코드

```python
# OpenCV 기능을 사용하기 위해 cv라는 이름으로 불러옵니다.
import cv2 as cv

# 수치 계산과 배열 처리를 위해 NumPy를 사용합니다.
import numpy as np

# 결과 이미지를 화면에 배치해서 보기 위해 pyplot을 불러옵니다.
import matplotlib.pyplot as plt

# 파일 경로를 다룰 때 사용할 os 모듈을 불러옵니다.
import os


# 현재 실행 중인 파이썬 파일의 위치를 기준으로 폴더 경로를 구합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))

# 현재 파일 위치를 기준으로 dabo.jpg의 전체 경로를 만듭니다.
image_path = os.path.join(base_dir, "../dabo.jpg")


# 지정한 경로에서 이미지를 읽어옵니다.
img = cv.imread(image_path)

# 이미지를 읽지 못한 경우
if img is None:
    # 어떤 경로에서 실패했는지 출력합니다.
    print(f"Error: Could not read image at {image_path}")
    # 더 이상 진행하지 않고 종료합니다.
    exit()


# matplotlib에서 색이 제대로 보이도록 BGR 이미지를 RGB로 바꿉니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 검출된 직선을 그릴 수 있도록 원본 이미지를 복사해 둡니다.
img_line = img.copy()


# 에지 검출 전에 컬러 이미지를 흑백 영상으로 변환합니다.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Canny 연산으로 이미지의 경계선을 추출합니다.
edges = cv.Canny(gray, threshold1=100, threshold2=200)


# 허프 변환을 이용해 에지 영상에서 직선 후보를 찾습니다.
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)


# 직선이 하나라도 검출된 경우
if lines is not None:
    # 검출된 각 직선을 하나씩 확인합니다.
    for line in lines:
        # 직선의 시작점과 끝점 좌표를 꺼냅니다.
        x1, y1, x2, y2 = line[0]
        # 복사한 이미지 위에 빨간색 선으로 직선을 그립니다.
        cv.line(img_line, (x1, y1), (x2, y2), (0, 0, 255), 2)


# matplotlib 출력용으로 직선이 그려진 이미지도 RGB로 변환합니다.
img_line_rgb = cv.cvtColor(img_line, cv.COLOR_BGR2RGB)


# 원본과 결과를 함께 보기 위해 figure 크기를 설정합니다.
plt.figure(figsize=(12, 6))


# 1행 2열 중 첫 번째 영역에 원본 이미지를 배치합니다.
plt.subplot(1, 2, 1)

# 원본 이미지를 화면에 출력합니다.
plt.imshow(img_rgb)

# 첫 번째 그림 제목을 설정합니다.
plt.title('Original Image')

# 축과 눈금은 보이지 않게 합니다.
plt.axis('off')


# 1행 2열 중 두 번째 영역에 결과 이미지를 배치합니다.
plt.subplot(1, 2, 2)

# 검출된 직선이 표시된 이미지를 출력합니다.
plt.imshow(img_line_rgb)

# 두 번째 그림 제목을 설정합니다.
plt.title('Detected Lines')

# 이 그림도 축은 숨깁니다.
plt.axis('off')


# 두 이미지의 간격이 겹치지 않도록 자동으로 정리합니다.
plt.tight_layout()

# 최종 결과 화면을 PNG 파일로 저장합니다.
plt.savefig('과제2_결과.png', dpi=300, bbox_inches='tight')

# figure를 닫아 메모리를 정리합니다.
plt.close()
```

## 주요 코드 설명

### 1. 그레이스케일 변환

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

직선 검출 전에 컬러 이미지를 흑백 영상으로 변환합니다.  
이렇게 하면 불필요한 색상 정보가 제거되어 에지 검출을 더 안정적으로 수행할 수 있습니다.

### 2. Canny 에지 검출

```python
edges = cv.Canny(gray, threshold1=100, threshold2=200)
```

Canny 알고리즘을 이용해 이미지의 경계선을 추출합니다.

- `threshold1` : 낮은 임계값  
- `threshold2` : 높은 임계값  

### 3. 허프 변환으로 직선 검출

```python
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
```

확률적 허프 변환을 사용해 에지 영상에서 직선을 찾습니다.

- `rho` : 거리 해상도  
- `theta` : 각도 해상도  
- `threshold` : 직선으로 판단할 최소 투표 수  
- `minLineLength` : 최소 직선 길이  
- `maxLineGap` : 끊어진 선을 이어줄 최대 간격  

### 4. 검출된 직선 그리기

```python
cv.line(img_line, (x1, y1), (x2, y2), (0, 0, 255), 2)
```

검출된 각 직선을 원본 이미지 위에 빨간색 선으로 표시합니다.

### 5. 결과 시각화

```python
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)

plt.subplot(1, 2, 2)
plt.imshow(img_line_rgb)
```

원본 이미지와 직선이 그려진 결과 이미지를 나란히 배치하여 비교합니다.

## 실행 결과
<img width="2137" height="856" alt="image" src="https://github.com/user-attachments/assets/b3e3ccc2-794c-4911-98b3-51a07957c1ea" />


---

# Practice 09: GrabCut을 이용한 객체 추출

## 전체 코드

```python
# OpenCV 기능을 사용하기 위해 cv라는 이름으로 불러옵니다.
import cv2 as cv

# 배열 처리와 수치 계산을 위해 NumPy를 불러옵니다.
import numpy as np

# 결과 이미지를 출력하기 위해 matplotlib의 pyplot을 사용합니다.
import matplotlib.pyplot as plt

# 파일 경로를 다루기 위해 os 모듈을 불러옵니다.
import os


# 현재 실행 중인 파일의 절대경로를 기준으로 폴더 위치를 구합니다.
base_dir = os.path.dirname(os.path.abspath(__file__))

# 사용할 이미지 파일의 전체 경로를 만듭니다.
image_path = os.path.join(base_dir, "../coffee cup.JPG")


# 지정한 경로에서 이미지를 읽어옵니다.
img = cv.imread(image_path)

# 이미지가 정상적으로 읽히지 않은 경우
if img is None:
    # 오류 메시지를 출력합니다.
    print(f"Error: Could not read image at {image_path}")
    # 프로그램 실행을 종료합니다.
    exit()


# matplotlib에서 올바른 색으로 보이도록 BGR 이미지를 RGB로 변환합니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)


# GrabCut에 사용할 초기 마스크를 0으로 생성합니다.
mask = np.zeros(img.shape[:2], np.uint8)

# 배경 모델을 저장할 배열을 생성합니다.
bgdModel = np.zeros((1, 65), np.float64)

# 전경 모델을 저장할 배열을 생성합니다.
fgdModel = np.zeros((1, 65), np.float64)


# 이미지의 높이와 너비를 가져옵니다.
h, w = img.shape[:2]

# 객체가 포함될 것으로 예상되는 초기 사각형 영역을 설정합니다.
rect = (w // 6, h // 6, w * 2 // 3, h * 2 // 3)


# 설정한 사각형을 기준으로 GrabCut 알고리즘을 수행합니다.
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)


# 배경으로 분류된 영역은 0, 전경으로 분류된 영역은 1로 바꿔 새로운 마스크를 만듭니다.
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')


# 새로 만든 마스크를 원본 이미지에 적용해 전경만 남깁니다.
img_extracted = img * mask2[:, :, np.newaxis]

# 결과 이미지를 matplotlib에서 보기 위해 RGB로 변환합니다.
img_extracted_rgb = cv.cvtColor(img_extracted, cv.COLOR_BGR2RGB)


# 원본, 마스크, 결과 이미지를 함께 보기 위해 figure 크기를 설정합니다.
plt.figure(figsize=(15, 6))


# 1행 3열 중 첫 번째 위치에 원본 이미지를 배치합니다.
plt.subplot(1, 3, 1)

# 원본 이미지를 출력합니다.
plt.imshow(img_rgb)

# 첫 번째 이미지 제목을 설정합니다.
plt.title('Original Image')

# 축은 보이지 않게 합니다.
plt.axis('off')


# 1행 3열 중 두 번째 위치에 마스크 이미지를 배치합니다.
plt.subplot(1, 3, 2)

# 생성한 마스크를 흑백 형태로 출력합니다.
plt.imshow(mask2, cmap='gray')

# 두 번째 이미지 제목을 설정합니다.
plt.title('Mask Image')

# 축은 숨깁니다.
plt.axis('off')


# 1행 3열 중 세 번째 위치에 배경이 제거된 결과 이미지를 배치합니다.
plt.subplot(1, 3, 3)

# 전경만 남은 이미지를 출력합니다.
plt.imshow(img_extracted_rgb)

# 세 번째 이미지 제목을 설정합니다.
plt.title('Extracted Object')

# 축은 보이지 않게 처리합니다.
plt.axis('off')


# 전체 배치를 보기 좋게 정리합니다.
plt.tight_layout()

# 결과 화면을 PNG 파일로 저장합니다.
plt.savefig('과제3_결과.png', dpi=300, bbox_inches='tight')

# figure를 닫아 메모리를 정리합니다.
plt.close()
```

## 주요 코드 설명

### 1. 초기 마스크와 모델 생성

```python
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
```

GrabCut 알고리즘을 수행하기 전에 필요한 초기 데이터를 준비합니다.

- `mask` : 각 픽셀이 배경인지 전경인지 저장하는 마스크  
- `bgdModel` : 배경 모델  
- `fgdModel` : 전경 모델  

### 2. 초기 사각형 설정

```python
rect = (w // 6, h // 6, w * 2 // 3, h * 2 // 3)
```

전경 객체가 포함될 것으로 예상되는 사각형 영역을 설정합니다.  
GrabCut은 이 사각형 내부에 객체가 있다고 가정하고 전경과 배경을 분리하기 시작합니다.

### 3. GrabCut 수행

```python
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)
```

지정한 사각형을 기준으로 GrabCut 알고리즘을 실행합니다.

- `iterCount=5` : 반복 횟수  
- `GC_INIT_WITH_RECT` : 사각형을 초기 조건으로 사용  

### 4. 전경 마스크 생성

```python
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
```

GrabCut 결과에서 배경으로 분류된 부분은 0, 전경으로 분류된 부분은 1로 바꾸어 새로운 이진 마스크를 만듭니다.

### 5. 객체 추출

```python
img_extracted = img * mask2[:, :, np.newaxis]
```

이진 마스크를 원본 이미지에 곱해서 전경만 남깁니다.

### 6. 결과 시각화

```python
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)

plt.subplot(1, 3, 2)
plt.imshow(mask2, cmap='gray')

plt.subplot(1, 3, 3)
plt.imshow(img_extracted_rgb)
```

원본 이미지, 마스크 이미지, 객체 추출 결과를 한 화면에 나란히 배치하여 비교합니다.

## 실행 결과
<img width="2139" height="577" alt="image" src="https://github.com/user-attachments/assets/f6b5f900-4505-42d7-a93c-99885fa758d1" />