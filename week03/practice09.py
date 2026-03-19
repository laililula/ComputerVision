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