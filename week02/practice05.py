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