# 1주차

## 01 이미지불러오기및그레이스케일변환
### 문제
<img width="1373" height="770" alt="image" src="https://github.com/user-attachments/assets/bcfc507a-3a5b-4079-af09-810b738a975c" />

### 코드              
01.grayscale.py
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
  
    • img_small = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5) : 이미지 크기 축소(fx : 가로 크기 비율, fy : 세로 크기 비율)  
    • gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) : 그레이스케일 변환  
    • gray_3 = cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR) : 채널 수 맞추기  
    • result = np.hstack((img_small, gray_3)) : 이미지 가로로 붙이기  

### 실행결과
<img width="2790" height="1024" alt="스크린샷 2026-03-05 144327" src="https://github.com/user-attachments/assets/f71de083-4d1d-4082-83e3-5cd2d9a56eca" />

## 02 페인팅붓크기조절기능추가
### 문제
<img width="1373" height="768" alt="image" src="https://github.com/user-attachments/assets/a262f1a5-cff3-48f6-861e-adbc55b26771" />


### 코드  
02.drawing.py
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
    • cv.setMouseCallback("Paint", draw) : 마우스 이벤트 발생 시 draw 함수 실행  
    • cv.circle(img, (x,y), brush_size, (255,0,0), -1) : 지정 위치에 원을 그려 그림 그리기 (brush_size : 붓 크기, (255,0,0) : 파란색)  
    • brush_size = min(15, brush_size+1) : 붓 크기 증가 (최대 15 제한)  
    • brush_size = max(1, brush_size-1) : 붓 크기 감소 (최소 1 제한)  
    • key = cv.waitKey(1) & 0xFF : 키보드 입력 확인  

### 실행결과
<img width="2866" height="1800" alt="스크린샷 2026-03-05 144502" src="https://github.com/user-attachments/assets/9d1a887a-7c29-4490-9e15-d652e5d6dafe" />


## 03 마우스로영역선택및ROI(관심영역) 추출
### 문제        
<img width="1374" height="771" alt="image" src="https://github.com/user-attachments/assets/27e213e7-f575-4ec4-b65d-fec839cde093" />

### 코드
03.shape_roi.py      
      
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
    • clone = img.copy() : 원본 이미지 보존을 위해 복사본 생성
    • x1, y1 = x, y : ROI 선택 시작 좌표 저장
    • cv.rectangle(img, (x1,y1), (x,y), (0,255,0), 2) : 드래그한 영역에 사각형 표시
    • roi = clone[y1:y, x1:x] : NumPy 슬라이싱을 이용하여 ROI 영역 추출
    • cv.imshow("ROI", roi) : 선택한 ROI 영역을 새로운 창에 출력
    • cv.imwrite("roi.png", roi) : 선택한 ROI 영역을 이미지 파일로 저장

### 실행결과
<img width="2856" height="1797" alt="image" src="https://github.com/user-attachments/assets/930e24a6-5786-4ba3-878b-3a43a4646f7a" />

