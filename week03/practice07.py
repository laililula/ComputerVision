# OpenCV 기능을 사용하기 위해 cv라는 이름으로 가져옵니다.
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
image_path = os.path.join(base_dir, "../edgeDetectionImage.jpg")


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


# 1행 2열 배치 중 첫 번째 위치를 선택합니다.
plt.subplot(1, 2, 1)

# 원본 이미지를 RGB 상태로 출력합니다.
plt.imshow(img_rgb)

# 첫 번째 그림 제목을 설정합니다.
plt.title('Original Image')

# 좌표축 표시를 없애서 이미지만 보이게 합니다.
plt.axis('off')


# 1행 2열 배치 중 두 번째 위치를 선택합니다.
plt.subplot(1, 2, 2)

# 계산한 에지 강도 영상을 회색조 형태로 출력합니다.
plt.imshow(magnitude, cmap='gray')

# 두 번째 그림 제목을 설정합니다.
plt.title('Edge Magnitude')

# 이쪽도 축은 보이지 않게 처리합니다.
plt.axis('off')


# 두 그림이 겹치지 않도록 간격을 자동으로 정리합니다.
plt.tight_layout()

# 현재 결과 화면을 PNG 파일로 저장합니다.
plt.savefig('과제1_결과.png', dpi=300, bbox_inches='tight')

# 사용한 figure 자원을 정리하고 창을 닫습니다.
plt.close()