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