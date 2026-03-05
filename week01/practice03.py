# 프로그램 종료 등을 위한 시스템 라이브러리
import sys

# OpenCV 라이브러리 (이미지 처리 및 마우스 이벤트 처리)
import cv2 as cv

# 배열 연산 라이브러리 (이미지는 numpy 배열 형태로 저장됨)
import numpy as np


# soccer.jpg 이미지를 읽어 img 변수에 저장
img = cv.imread('soccer.jpg')

# 이미지 파일이 존재하지 않으면 프로그램 종료
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')


# ROI 선택을 초기 상태로 되돌리기 위해 원본 이미지를 복사
clone = img.copy()

# ROI 선택 시작 좌표를 저장할 변수
start_point = None

# 현재 마우스 드래그 중인지 여부를 저장하는 변수
drawing = False

# 선택된 ROI 이미지를 저장할 변수
roi = None


# 마우스 이벤트 발생 시 실행되는 함수 정의
def select_roi(event, x, y, flags, param):

    # 함수 내부에서 전역 변수 사용을 위해 선언
    global start_point, drawing, img, clone, roi

    # 마우스 좌클릭을 눌렀을 때 ROI 선택 시작
    if event == cv.EVENT_LBUTTONDOWN:

        # 드래그 시작 상태로 변경
        drawing = True

        # ROI 시작 좌표를 현재 마우스 위치로 설정
        start_point = (x, y)

    # 마우스를 움직일 때 발생하는 이벤트
    elif event == cv.EVENT_MOUSEMOVE:

        # 드래그 중일 때만 실행
        if drawing:

            # 기존 사각형을 지우기 위해 원본 이미지를 복사
            img = clone.copy()

            # 시작점부터 현재 마우스 위치까지 사각형을 그려 ROI 영역을 표시
            cv.rectangle(img, start_point, (x, y), (0, 255, 0), 2)

    # 마우스 좌클릭 버튼을 놓았을 때 ROI 선택 완료
    elif event == cv.EVENT_LBUTTONUP:

        # 드래그 상태 종료
        drawing = False

        # 시작 좌표 저장
        x0, y0 = start_point

        # 현재 마우스 좌표 저장
        x1, y1 = x, y

        # 드래그 방향에 상관없이 좌표 정렬
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)

        # numpy 슬라이싱을 이용해 선택된 ROI 영역을 추출
        roi = clone[y_min:y_max, x_min:x_max]

        # ROI가 비어있지 않으면 별도의 창에 출력
        if roi.size != 0:
            cv.imshow("ROI", roi)


# Image라는 이름의 OpenCV 창 생성
cv.namedWindow("Image")

# 마우스 이벤트 발생 시 select_roi 함수가 호출되도록 등록
cv.setMouseCallback("Image", select_roi)


# 프로그램이 계속 실행되도록 무한 반복 루프 시작
while True:

    # 현재 이미지 상태를 화면에 출력
    cv.imshow("Image", img)

    # 1ms 동안 키 입력을 대기하고 입력값 저장
    key = cv.waitKey(1) & 0xFF

    # q 키를 누르면 프로그램 종료
    if key == ord('q'):
        break

    # r 키를 누르면 ROI 선택 상태 초기화
    elif key == ord('r'):

        # 원본 이미지로 복원
        img = clone.copy()

        # ROI 변수 초기화
        roi = None

    # s 키를 누르면 ROI 이미지 저장
    elif key == ord('s'):

        # ROI가 존재할 경우에만 저장
        if roi is not None:

            # ROI 이미지를 파일로 저장
            cv.imwrite("roi.png", roi)

            # 저장 완료 메시지 출력
            print("ROI 이미지가 roi.png로 저장되었습니다.")


# 프로그램 종료 시 모든 OpenCV 창을 닫음
cv.destroyAllWindows()