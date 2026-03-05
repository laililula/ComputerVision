# Week01 Practice README

이 README 파일은 week01 폴더의 practice01.py, practice02.py, practice03.py 파일들의 전체 코드, 주요 코드 설명, 그리고 실행 결과를 포함합니다.

## Practice 01: 이미지 흑백 변환 및 표시

### 전체 코드
```python
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
```

## 주요 코드 설명

### 1. 이미지 불러오기

```python
img = cv.imread('soccer.jpg')
```

OpenCV의 `cv.imread()` 함수를 사용하여 이미지를 읽어옵니다.

- 이미지 파일을 **NumPy 배열 형태**로 저장합니다.
- OpenCV에서 컬러 이미지는 **BGR 형식**으로 저장됩니다.
- 파일이 존재하지 않으면 `None`이 반환됩니다.

---

### 2. 이미지 존재 여부 확인

```python
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
```

이미지 파일이 정상적으로 로드되었는지 확인합니다.

- 파일이 없으면 프로그램을 종료합니다.
- `sys.exit()`을 사용하여 오류 메시지를 출력합니다.
- 이 과정을 통해 **잘못된 경로로 인한 오류를 방지할 수 있습니다.**

---

### 3. 그레이스케일 변환

```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

컬러 이미지를 **흑백 이미지(Grayscale)** 로 변환합니다.

- `cv.cvtColor()` : 색상 공간을 변환하는 함수  
- `cv.COLOR_BGR2GRAY` : **BGR → Gray 변환 옵션**

Grayscale 이미지는 **1채널 이미지**이며  
**밝기 정보만 포함합니다.**

---

### 4. Gray → BGR 변환

```python
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
```

흑백 이미지를 다시 **3채널 이미지(BGR)** 로 변환합니다.

이 과정이 필요한 이유

- 컬러 이미지 : **3채널**
- 그레이 이미지 : **1채널**

채널 수가 다르면 `np.hstack()`으로  
이미지를 연결할 수 없습니다.

따라서 **채널을 맞추기 위해 변환합니다.**

---

### 5. 이미지 가로 연결

```python
combined = np.hstack((img, gray_bgr))
```

NumPy의 `hstack()` 함수를 이용하여  
두 이미지를 **가로 방향으로 연결합니다.**

결과 구조

```
[컬러 이미지 | 그레이 이미지]
```

이렇게 하면 두 이미지를 **한 화면에서 비교할 수 있습니다.**

---

### 6. 이미지 크기 축소

```python
combined = cv.resize(combined, None, fx=0.5, fy=0.5)
```

이미지 크기를 **50%로 축소합니다.**

- `fx` : 가로 비율
- `fy` : 세로 비율

이미지를 축소하는 이유

- 화면에 출력하기 쉽게 만들기 위해서
- 너무 큰 이미지 출력 방지

---

### 7. 이미지 출력

```python
cv.imshow('Original vs Gray', combined)
```

OpenCV 창을 생성하고 이미지를 화면에 출력합니다.

- `'Original vs Gray'` : 창 이름
- `combined` : 출력할 이미지

---

### 8. 키 입력 대기

```python
cv.waitKey(0)
```

사용자가 **키를 누를 때까지 프로그램을 대기합니다.**

- `0` → 무한 대기
- 키를 누르면 다음 코드 실행

---

### 9. OpenCV 창 종료

```python
cv.destroyAllWindows()
```

열려 있는 **모든 OpenCV 창을 종료합니다.**

프로그램 종료 시  
남아있는 창을 정리하는 역할을 합니다.

### 실행 결과
<img width="2866" height="1014" alt="image" src="https://github.com/user-attachments/assets/06baf2e7-cf8a-4987-8693-f2bc5e1d68ef" />

## Practice 02: 마우스로 그림 그리기

### 전체 코드
```python
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
```

## 주요 코드 설명

### 1. 붓 크기 및 그리기 상태 변수 설정

```python
brush_size = 5
drawing = False
color = (255, 0, 0)
```

마우스를 이용한 그림 그리기에 필요한 **기본 상태 변수**를 설정합니다.

- `brush_size` : 붓의 크기를 저장하는 변수
- `drawing` : 마우스를 누른 상태에서 드래그 중인지 여부를 저장
- `color` : 붓의 색상을 저장 (BGR 기준)

기본값은 **파란색 붓**으로 설정됩니다.

---

### 2. 마우스 이벤트 처리 함수 정의

```python
def draw(event, x, y, flags, param):
```

마우스 이벤트가 발생할 때 실행되는 **콜백 함수**입니다.

OpenCV는 마우스 이벤트 발생 시 다음 정보를 전달합니다.

- `event` : 발생한 마우스 이벤트 종류
- `x, y` : 마우스 좌표
- `flags` : 이벤트 상태 정보
- `param` : 추가 파라미터

---

### 3. 좌클릭 시 그리기 시작 (파란색)

```python
if event == cv.EVENT_LBUTTONDOWN:
    drawing = True
    color = (255, 0, 0)
```

마우스 **좌클릭 버튼을 누르면** 다음 동작을 수행합니다.

- 드래그 상태(`drawing`)를 **True로 변경**
- 붓 색상을 **파란색(BGR)** 으로 설정

이 상태에서 마우스를 움직이면 그림이 그려집니다.

---

### 4. 우클릭 시 그리기 시작 (빨간색)

```python
elif event == cv.EVENT_RBUTTONDOWN:
    drawing = True
    color = (0, 0, 255)
```

마우스 **우클릭 버튼을 누르면**

- 드래그 상태를 **True로 변경**
- 붓 색상을 **빨간색(BGR)** 으로 설정

---

### 5. 마우스 이동 시 그림 그리기

```python
elif event == cv.EVENT_MOUSEMOVE:
    if drawing:
        cv.circle(img, (x, y), brush_size, color, -1)
```

마우스를 이동할 때 발생하는 이벤트입니다.

- `drawing` 상태일 때만 그림을 그립니다.
- 현재 마우스 위치 `(x, y)`에 **원을 그려 붓 효과를 생성**합니다.

`cv.circle()` 함수

- `(x, y)` : 중심 좌표
- `brush_size` : 원의 반지름
- `color` : 붓 색상
- `-1` : 원 내부를 채움

---

### 6. 마우스 버튼을 놓으면 그리기 종료

```python
elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
    drawing = False
```

마우스 버튼을 놓으면

- 드래그 상태를 **False로 변경**
- 그림 그리기가 종료됩니다.

---

### 7. 마우스 이벤트 등록

```python
cv.namedWindow('Paint')
cv.setMouseCallback('Paint', draw)
```

OpenCV 창에 **마우스 이벤트를 연결**합니다.

- `cv.namedWindow()` : 창 생성
- `cv.setMouseCallback()` : 마우스 이벤트 발생 시 실행할 함수 등록

즉 **Paint 창에서 발생하는 마우스 이벤트가 `draw()` 함수로 전달됩니다.**

---

### 8. 실시간 화면 갱신을 위한 반복문

```python
while True:
```

프로그램이 종료될 때까지 **무한 반복 루프**를 실행합니다.

이 루프를 통해

- 화면을 계속 업데이트
- 마우스 입력
- 키보드 입력

을 실시간으로 처리할 수 있습니다.

---

### 9. 키보드 입력 처리

```python
key = cv.waitKey(1) & 0xFF
```

키보드 입력을 1ms 동안 대기하고 값을 저장합니다.

`& 0xFF` 연산은 **플랫폼에 따라 발생하는 키 값 차이를 제거하기 위해 사용됩니다.**

---

### 10. 프로그램 종료 키

```python
if key == ord('q'):
    break
```

'q' 키를 누르면

- 반복문을 종료하고
- 프로그램이 종료됩니다.

---

### 11. 붓 크기 증가

```python
elif key == ord('+'):
    brush_size = min(15, brush_size + 1)
```

'+' 키를 누르면 **붓 크기가 증가합니다.**

- 최대 크기는 **15**로 제한됩니다.

---

### 12. 붓 크기 감소

```python
elif key == ord('-'):
    brush_size = max(1, brush_size - 1)
```

'-' 키를 누르면 **붓 크기가 감소합니다.**

- 최소 크기는 **1**로 제한됩니다.

### 실행 결과
<img width="2879" height="1700" alt="image" src="https://github.com/user-attachments/assets/17277542-9d76-4609-ae3a-cdf5177fa2e8" />


## Practice 03: ROI 선택 및 저장

### 전체 코드
```python
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
```

## 주요 코드 설명

### 1. 원본 이미지 복사

```python
clone = img.copy()
```

ROI 선택 과정에서 **이미지 위에 사각형을 계속 그리기 때문에 원본 이미지를 보존하기 위해 복사본을 생성**합니다.

- `clone` : 원본 이미지를 저장하는 변수
- ROI 선택 중 사각형을 다시 그릴 때 **기존 사각형을 제거하기 위해 사용됩니다.**

---

### 2. ROI 선택 상태 변수

```python
start_point = None
drawing = False
roi = None
```

ROI 선택을 위해 필요한 상태 변수들입니다.

- `start_point` : ROI 선택 시작 좌표 저장
- `drawing` : 마우스 드래그 중인지 여부
- `roi` : 선택된 ROI 이미지를 저장하는 변수

---

### 3. 마우스 이벤트 처리 함수

```python
def select_roi(event, x, y, flags, param):
```

마우스 이벤트가 발생할 때 실행되는 **콜백 함수**입니다.

OpenCV는 마우스 이벤트 발생 시 다음 정보를 전달합니다.

- `event` : 발생한 마우스 이벤트 종류
- `x, y` : 마우스 좌표
- `flags` : 이벤트 상태
- `param` : 추가 전달 파라미터

---

### 4. ROI 선택 시작

```python
if event == cv.EVENT_LBUTTONDOWN:
    drawing = True
    start_point = (x, y)
```

마우스 **좌클릭 버튼을 누르면 ROI 선택이 시작됩니다.**

- `drawing = True` → 드래그 상태 시작
- 현재 마우스 좌표 `(x, y)`를 **ROI 시작 좌표**로 저장

---

### 5. 드래그 중 ROI 영역 표시

```python
elif event == cv.EVENT_MOUSEMOVE:
    if drawing:
        img = clone.copy()
        cv.rectangle(img, start_point, (x, y), (0, 255, 0), 2)
```

마우스를 드래그하면 **ROI 선택 영역을 사각형으로 표시**합니다.

동작 과정

1. `img = clone.copy()`  
   → 기존 사각형을 지우기 위해 원본 이미지로 복원

2. `cv.rectangle()`  
   → 시작 좌표부터 현재 마우스 위치까지 **초록색 사각형 표시**

`cv.rectangle()` 주요 파라미터

- `start_point` : 시작 좌표
- `(x, y)` : 현재 마우스 좌표
- `(0,255,0)` : 초록색
- `2` : 선 두께

---

### 6. ROI 선택 완료

```python
elif event == cv.EVENT_LBUTTONUP:
    drawing = False
```

마우스 버튼을 놓으면

- 드래그 상태 종료
- ROI 선택이 완료됩니다.

---

### 7. 좌표 정렬

```python
x0, y0 = start_point
x1, y1 = x, y

x_min, x_max = min(x0, x1), max(x0, x1)
y_min, y_max = min(y0, y1), max(y0, y1)
```

마우스를 **어떤 방향으로 드래그해도 ROI가 정상적으로 선택되도록 좌표를 정렬**합니다.

예를 들어

```
왼쪽 → 오른쪽
오른쪽 → 왼쪽
위 → 아래
아래 → 위
```

모든 경우에서 올바른 ROI 영역을 만들 수 있습니다.

---

### 8. ROI 영역 추출

```python
roi = clone[y_min:y_max, x_min:x_max]
```

NumPy 슬라이싱을 이용하여 **선택한 영역의 이미지를 추출**합니다.

이미지는 NumPy 배열이기 때문에

```
image[세로 범위, 가로 범위]
```

형태로 ROI를 추출할 수 있습니다.

---

### 9. ROI 이미지 출력

```python
if roi.size != 0:
    cv.imshow("ROI", roi)
```

선택된 ROI 영역이 존재하면

- 별도의 **ROI 창을 생성하여 이미지 출력**

`roi.size != 0` 조건을 통해  
**빈 영역 선택 오류를 방지**합니다.

---

### 10. 마우스 이벤트 등록

```python
cv.namedWindow("Image")
cv.setMouseCallback("Image", select_roi)
```

OpenCV 창에 **마우스 이벤트를 연결**합니다.

- `cv.namedWindow()` → 창 생성
- `cv.setMouseCallback()` → 마우스 이벤트 처리 함수 등록

즉 **Image 창에서 발생하는 마우스 이벤트가 `select_roi()` 함수로 전달됩니다.**

---

### 11. ROI 초기화 기능

```python
elif key == ord('r'):
    img = clone.copy()
    roi = None
```

`r` 키를 누르면

- 이미지가 **원본 상태로 복원**
- 기존 ROI 선택이 **초기화**

---

### 12. ROI 이미지 저장

```python
elif key == ord('s'):
    if roi is not None:
        cv.imwrite("roi.png", roi)
```

`s` 키를 누르면 선택한 ROI 이미지를 **파일로 저장**합니다.

- 파일 이름 : `roi.png`
- `cv.imwrite()` 함수 사용

---

### 13. 저장 완료 메시지 출력

```python
print("ROI 이미지가 roi.png로 저장되었습니다.")
```

ROI 저장이 완료되면 **터미널에 메시지를 출력**하여 사용자에게 저장 여부를 알려줍니다.

### 실행 결과
<img width="2676" height="1604" alt="image" src="https://github.com/user-attachments/assets/92909417-a0df-49b4-9213-50e36e48e081" />
<img width="382" height="518" alt="image" src="https://github.com/user-attachments/assets/9279be20-1bb3-4b66-8eb9-c131e0b03362" />

