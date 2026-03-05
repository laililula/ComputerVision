# 프로그램 종료 등을 위한 시스템 라이브러리
import sys
# OpenCV 라이브러리 (이미지 처리)
import cv2 as cv
# 배열 연산 라이브러리 (이미지는 numpy 배열로 저장됨)
import numpy as np

# 이미지 읽기
# soccer.jpg 파일을 읽어 컬러 이미지(BGR)로 저장
img = cv.imread('soccer.jpg')

# 이미지 파일이 없으면 프로그램 종료
if img is None:                 
    sys.exit('파일을 찾을 수 없습니다.')

# Gray 변환 : 컬러(BGR) 이미지를 흑백 이미지로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Gray → BGR 변환 (채널 맞추기) : 흑백 이미지를 3채널(BGR)로 변환 (이미지 연결을 위해)
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 가로 연결 : 컬러 이미지와 흑백 이미지를 가로로 붙임
combined = np.hstack((img, gray_bgr))

# 이미지 크기를 50%로 축소
combined = cv.resize(combined, None, fx=0.5, fy=0.5)  

# 연결된 이미지를 화면에 출력
cv.imshow('Original vs Gray', combined) 

# 키 입력이 있을 때까지 창 유지
cv.waitKey(0)         

# 모든 OpenCV 창 닫기   
cv.destroyAllWindows()