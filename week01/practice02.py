# 프로그램 종료 등을 위한 시스템 라이브러리
import sys

# OpenCV 라이브러리 (이미지 처리 및 마우스 이벤트 처리)
import cv2 as cv

# 배열 연산 라이브러리 (이미지는 numpy 배열 형태로 저장됨)
import numpy as np


# soccer.jpg 이미지를 읽어서 img 변수에 저장
img = cv.imread('soccer.jpg')

# 이미지 파일이 존재하지 않으면 프로그램 종료
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')


# 초기 붓 크기를 5로 설정
brush_size = 5

# 마우스를 누르고 드래그 중인지 여부를 저장하는 변수
drawing = False

# 기본 붓 색상 설정 (BGR 기준 파란색)
color = (255, 0, 0)


# 마우스 이벤트가 발생할 때 실행되는 함수 정의
def draw(event, x, y, flags, param):

    # 함수 내부에서 전역 변수 사용을 위해 선언
    global drawing, color, brush_size

    # 마우스 좌클릭 버튼을 눌렀을 때 실행
    if event == cv.EVENT_LBUTTONDOWN:

        # 드래그 시작 상태로 변경
        drawing = True

        # 붓 색상을 파란색으로 설정
        color = (255, 0, 0)

    # 마우스 우클릭 버튼을 눌렀을 때 실행
    elif event == cv.EVENT_RBUTTONDOWN:

        # 드래그 시작 상태로 변경
        drawing = True

        # 붓 색상을 빨간색으로 설정
        color = (0, 0, 255)

    # 마우스를 움직일 때 발생하는 이벤트
    elif event == cv.EVENT_MOUSEMOVE:

        # 드래그 상태일 때만 그림을 그림
        if drawing:

            # 현재 마우스 위치에 붓 크기의 원을 그려 그림을 생성
            cv.circle(img, (x, y), brush_size, color, -1)

    # 마우스 좌클릭 또는 우클릭 버튼을 놓았을 때 실행
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:

        # 드래그 상태 종료
        drawing = False


# Paint라는 이름의 OpenCV 창 생성
cv.namedWindow('Paint')

# 마우스 이벤트가 발생하면 draw 함수가 호출되도록 등록
cv.setMouseCallback('Paint', draw)


# 프로그램이 계속 실행되도록 무한 반복 루프 시작
while True:

    # 현재 이미지 상태를 Paint 창에 출력
    cv.imshow('Paint', img)

    # 1ms 동안 키보드 입력을 대기하고 입력값을 저장
    key = cv.waitKey(1) & 0xFF

    # q 키를 누르면 프로그램 종료
    if key == ord('q'):
        break

    # + 키를 누르면 붓 크기를 1 증가 (최대 15까지)
    elif key == ord('+'):
        brush_size = min(15, brush_size + 1)

    # - 키를 누르면 붓 크기를 1 감소 (최소 1까지)
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)


# 프로그램 종료 시 모든 OpenCV 창을 닫음
cv.destroyAllWindows()