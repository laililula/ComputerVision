import sys
import cv2 as cv

img = cv.imread('soccer.jpg')   # 영상 읽기

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.imshow('Image Display', img) # 윈도우에 영상 표시 ('창 이름', 표시할 이미지)

cv.waitKey()
cv.destroyAllWindows()