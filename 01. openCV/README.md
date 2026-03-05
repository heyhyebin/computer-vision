# 1주차

## 01 이미지불러오기및그레이스케일변환
### 문제
<img width="1373" height="770" alt="image" src="https://github.com/user-attachments/assets/bcfc507a-3a5b-4079-af09-810b738a975c" />

### 코드              
grayscal.py
```python
import cv2 as cv    # OpenCV 라이브러리를 cv라는 이름으로 불러오기
import sys  # 프로그램 종료를 위해 sys 모듈 사용
import numpy as np

img = cv.imread('girl_laughing.jpg')  # 이미지 읽기
img_small=cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)   # 이미지 사이즈 반으로 축소

# 이미지 파일이 없을 경우
if img is None:
  sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일 변환
gray_small=cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5) # 이미지 사이즈 반으로 축소

gray_3 = cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR) # gray를 3채널로 변환 (hstack 맞추기 위해)

result = np.hstack((img_small, gray_3)) # 가로로 붙이기

# 출력
cv.imshow("Original vs Gray", result)
cv.waitKey()
cv.destroyAllWindows()
```




### 핵심 코드 설명

1. 이미지 읽기  
```python
img = cv.imread('girl_laughing.jpg')
```
  OpenCV의 imread() 함수를 사용하여 이미지를 불러온다.

2. 이미지 크기 축소  
```python
img_small = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)  
```
  resize() 함수를 이용하여 이미지 크기를 절반으로 줄인다.    
      • fx : 가로 크기 비율  
      • fy : 세로 크기 비율

3. 그레이스케일 변환
```python
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```
4. 채널 수 맞추기
```python
  gray_3 = cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR)
```
  np.hstack()으로 이미지를 붙이기 위해서는 이미지 채널 수가 같아야 한다.  
    • 컬러 이미지 → 3채널  
    • 그레이스케일 → 1채널

5. 이미지 가로로 붙이기
```python
  result = np.hstack((img_small, gray_3))
```

### 실행결과
<img width="2790" height="1024" alt="스크린샷 2026-03-05 144327" src="https://github.com/user-attachments/assets/f71de083-4d1d-4082-83e3-5cd2d9a56eca" />

## 02 페인팅붓크기조절기능추가
### 문제
<img width="1373" height="768" alt="image" src="https://github.com/user-attachments/assets/a262f1a5-cff3-48f6-861e-adbc55b26771" />


### 코드  
drawing.py
```python
import cv2 as cv    # OpenCV 라이브러리를 cv라는 이름으로 불러오기
import sys  # 프로그램 종료를 위해 sys 모듈 사용
import numpy as np

img = cv.imread('girl_laughing.jpg')    # 이미지 읽기
# 이미지 파일이 없을 경우
if img is None:
  sys.exit('파일이 존재하지 않습니다.')

brush_size = 5  # 초기 붓 크기 지정

def draw(event, x, y, flags, param):  # 콜백 함수
    global brush_size
    # 좌클릭하여 마우스를 움직이는 경우 원으로 파란색으로 그리기
    if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON):
        cv.circle(img, (x,y), brush_size, (255,0,0), -1)
    # 우클릭하여 마우스를 움직이는 경우 원으로 빨간색으로 그리기
    if event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON):
        cv.circle(img, (x,y), brush_size, (0,0,255), -1)

cv.namedWindow("Paint")
cv.setMouseCallback("Paint", draw)

while True:
    cv.imshow("Paint", img)
    key = cv.waitKey(1) & 0xFF  # 키 입력 확인

    if key == ord('+'):   # + 입력한 경우 붓 크기 증가
        brush_size = min(15, brush_size+1)

    elif key == ord('-'): # - 입력한 경우 붓 크기 감소
        brush_size = max(1, brush_size-1)

    elif key == ord('q'): # 창 종료
        break

cv.destroyAllWindows()
```




### 핵심 코드 설명
1. 좌클릭으로 파란색 그림 그리기
```python
   if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON):
    cv.circle(img, (x,y), brush_size, (255,0,0), -1)
```
왼쪽 마우스를 누르거나 누른 상태에서 움직이면 파란색 원을 그린다.
cv.circle(이미지, 중심좌표, 반지름, 색상, 두께)
    • (x,y) : 마우스 위치
    • brush_size : 원의 크기
    • (255,0,0) : 파란색 (BGR)
    • -1 : 내부를 채운 원

2. 키보드 입력 처리
```python
key = cv.waitKey(1) & 0xFF
```
키보드 입력을 확인하기 위한 코드이다.

3. 붓 크기 증가  
```python
if key == ord('+'):
    brush_size = min(15, brush_size+1)
```
  + 키를 누르면 붓 크기가 증가한다.
  최대 크기는 15로 제한된다.

### 실행결과
<img width="2866" height="1800" alt="스크린샷 2026-03-05 144502" src="https://github.com/user-attachments/assets/9d1a887a-7c29-4490-9e15-d652e5d6dafe" />


## 03 마우스로영역선택및ROI(관심영역) 추출
### 문제        
<img width="1374" height="771" alt="image" src="https://github.com/user-attachments/assets/27e213e7-f575-4ec4-b65d-fec839cde093" />

### 코드
shape.py      
      
```python
import cv2 as cv    # OpenCV 라이브러리를 cv라는 이름으로 불러오기
import sys  # 프로그램 종료를 위해 sys 모듈 사용

img = cv.imread('girl_laughing.jpg')    # 이미지 읽기
# 이미지 파일이 없을 경우
if img is None:
  sys.exit('파일이 존재하지 않습니다.')

clone = img.copy()  # 원본 이미지를 복사해서 clone에 저장 (원본 보존용)

x1, y1 = -1, -1 # 드래그할 마우스 초기 위치
roi = None

def select_roi(event, x, y, flags, param):  # 콜백 함수
    global x1, y1, img, clone, roi

    if event == cv.EVENT_LBUTTONDOWN:   # 왼쪽 마우스를 눌렸을 때
        x1, y1 = x, y   # 드래그 시작 좌표

    elif event == cv.EVENT_LBUTTONUP:   # 왼쪽 마우스 드래그 했을 때
        cv.rectangle(img, (x1,y1), (x,y), (0,255,0), 2) # 사각형 그리기

        # ROI 영역 추출 (numpy 슬라이싱)
        # y좌표 먼저, x좌표 
        roi = clone[y1:y, x1:x]
        cv.imshow("ROI", roi)

cv.namedWindow("image")
cv.setMouseCallback("image", select_roi)

while True:

    cv.imshow("image", img)
    key = cv.waitKey(1) & 0xFF

    if key == ord('r'):     # r 입력한 경우 초기화
        img = clone.copy()

    elif key == ord('s') and roi is not None:   # s 입력한 경우 지정된 영역 저장
        cv.imwrite("roi.png", roi)

    elif key == ord('q'):   # 창 종료
        break

cv.destroyAllWindows()
```

### 핵심 코드 설명
1. 원본 이미지 복사
```python
clone = img.copy()
```
이미지의 원본을 보존하기 위해 복사본을 생성한다.
ROI를 선택하거나 그림을 그릴 때 원본 이미지를 유지하기 위해 사용한다.

2. 드래그 시작 좌표 저장
```python
if event == cv.EVENT_LBUTTONDOWN:
    x1, y1 = x, y
```
왼쪽 마우스를 누르면 ROI 선택의 시작 좌표를 저장한다.

3. 사각형 그리기
```python
elif event == cv.EVENT_LBUTTONUP:
    cv.rectangle(img, (x1,y1), (x,y), (0,255,0), 2)
```
마우스를 떼면 시작 좌표 (x1,y1)와 현재 좌표 (x,y)를 이용해 이미지 위에 사각형을 그린다.

cv.rectangle(이미지, 시작좌표, 끝좌표, 색상, 두께)

4. ROI 영역 추출
```python
roi = clone[y1:y, x1:x]
```
ROI는 Region Of Interest로, 이미지에서 관심 영역을 의미한다.
OpenCV 이미지는 NumPy 배열이기 때문에 슬라이싱을 사용하여 영역을 추출할 수 있다.
      • 이미지[y좌표, x좌표]

5. 선택된 영역 출력
```python
cv.imshow("ROI", roi)
```
선택된 ROI 영역을 새로운 창에 출력한다.
6. 키 (ROI 저장)
```python
elif key == ord('s') and roi is not None:
    cv.imwrite("roi.png", roi)
```
선택된 ROI 영역을 파일로 저장한다.

### 실행결과
<img width="2879" height="1699" alt="스크린샷 2026-03-05 144713" src="https://github.com/user-attachments/assets/bce7a51c-e70f-415a-bb8d-8a439b9978fc" />
