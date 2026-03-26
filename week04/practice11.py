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