import sys
import cv2 as cv

img = cv.imread('soccer.jpg')                               # 영상 읽기

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                  # BGR 컬러 영상을 명암 영상으로 변환
gray_small = cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)   # 반으로 축소

cv.imwrite('soccer.jpg', gray)

cv.imshow('Image Display', img)                             # 윈도우에 영상 표시 ('창 이름', 표시할 이미지)

cv.waitKey()
cv.destroyAllWindows()