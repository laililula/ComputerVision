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