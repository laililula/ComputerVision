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