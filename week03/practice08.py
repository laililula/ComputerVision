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