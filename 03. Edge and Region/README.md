# 3주차

## 01. 소벨 에지 검출 및 결과 시각화

### 문제
#### 설명
  • edgeDetectionImage 이미지를 그레이스케일로 변환   
  • Sobel 필터를 사용하여 x축과 y축 방향의 에지를 검출            
  • 검출된 에지 강도 이미지를 시각화   

#### 요구사항
  • cv.imread()를 사용하여 이미지를 불러옴                 
  • cv.cvtColor()를 사용하여 그레이스케일로변환                 
  • cvSobel()을 사용하여 x축(cv.CV_64F, 1, 0)과 y축(cv.CV_64F, 0,1) 방향의 에지를 검출                 
  • cv.magnitude()를 사용하여 에지강도계산                 
  • Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화                 

#### 힌트
  • cv.Sobel()의 ksize는 3 또는 5로 설정                 
  • cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 uint8로 변환                 
  • plt.imshow()에서 cmap=‘gray’를 사용하여 흑백으로 시각화                 

### 코드
01.sobel.py
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img = cv.imread('edgeDetectionImage.jpg')

if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 2. 그레이스케일 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 3. Sobel 필터로 x, y 방향 에지 검출
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)    # 세로 방향 변화 → 수직 경계 검출
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)    # 가로 방향 변화 → 수직 경계 검출

# 4. 에지 강도 계산 : 두 방향의 그래디언트를 결합하여 전체 에지 강도 계산
magnitude = cv.magnitude(sobel_x, sobel_y)

# 5. uint8 형식으로 변환 : 시각화를 위해 0~255 범위로 변환
magnitude_abs = cv.convertScaleAbs(magnitude)

# 6. BGR -> RGB 변환 (matplotlib 출력용)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 7. 결과 시각화
plt.figure(figsize=(12, 6))

# 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 에지 강도 이미지 출력
plt.subplot(1, 2, 2)
plt.imshow(magnitude_abs, cmap='gray')
plt.title('Sobel Edge Magnitude')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 핵심 코드
    • sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
      x 방향 밝기 변화(수직 경계)를 계산하는 코드
      
    • sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
      y 방향 밝기 변화(수평 경계)를 계산하는 코드
      
    • magnitude = cv.magnitude(sobel_x, sobel_y)
      x, y 에지를 합쳐 전체 에지 강도를 계산하는 코드

### 실행결과
<img width="2404" height="1324" alt="스크린샷 2026-03-19 133659" src="https://github.com/user-attachments/assets/b5f1710b-c50f-4729-b309-e3f072d4c3e8" />




## 02. 캐니 에지 및 허프 변환을 이용한 직선 검출

### 문제
#### 설명
  • dabo 이미지에 캐니 에지 검출을 사용하여 에지 맵 생성               
  • 허프 변환을 사용하여 이미지에서 직선 검출               
  • 검출된 직선을 원본 이미지에서 빨간색으로 표시               
  
#### 요구사항
  • cv.Canny()를 사용하여 에지 맵 생성               
  • cv.HoughtLinesP()를 사용하여 직선 검출               
  • cv.line()을 사용하여 검출된 직선을 원본 이미지에 그림               
  • Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화               
  
#### 힌트
  • Cv.Canny()에서 threshold1과 threshold2는 100과 200으로 설정               
  • cv.HoughLinesP()에서 rho, theta, threshold, minLineLength, maxLineGap 값을 조정하여 직선 검출 성능을 개선               
  • cv.line()에서 색상은 (0, 0, 255) (빨간색)과 두께는 2로 설정               

### 코드
02.line_detection.py
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img = cv.imread('dabo.jpg')

if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 원본 복사
line_img = img.copy()

# 2. 그레이스케일 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 노이즈 제거
gray = cv.GaussianBlur(gray, (5, 5), 0)

# 3. 캐니 에지 검출
# threshold1: 낮은 임계값, threshold2: 높은 임계값
edges = cv.Canny(gray, 100, 200)

# 4. 허프 변환으로 직선 검출
lines = cv.HoughLinesP(
    edges,                 # 입력 에지 이미지
    rho=1,                 # 거리 해상도 (픽셀 단위)
    theta=np.pi / 180,     # 각도 해상도 (라디안)
    threshold=120,         # 직선으로 인정할 최소 투표 수
    minLineLength=60,      # 최소 직선 길이
    maxLineGap=10          # 직선 사이 최대 간격
)

# 5. 검출된 직선을 원본 이미지에 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)   # (0, 0, 255) = 빨간색, 두께 2

# 6. BGR -> RGB 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)

# 7. 결과 시각화
plt.figure(figsize=(12, 6))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 직선 검출 결과
plt.subplot(1, 2, 2)
plt.imshow(line_img_rgb)
plt.title('Detected Lines')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 핵심 코드
    • edges = cv.Canny(gray, 100, 200)
      이미지에서 경계를 검출하여 에지 맵을 생성하는 코드
    
    • lines = cv.HoughLinesP(edges, 1, np.pi/180, 120, minLineLength=60, maxLineGap=10)
      에지 이미지에서 직선 성분을 검출하는 코드
    
    • cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
      검출된 직선을 원본 이미지 위에 시각적으로 표시하는 코드

### 실행결과
<img width="2404" height="1324" alt="image" src="https://github.com/user-attachments/assets/758fb8cc-b0d7-4788-9b04-c66148270fc6" />





## 03. GrabCut을 이용한 대화식 영역 분할 및 객체 추출

### 문제
#### 설명
  • coffee cup 이미지로 사용자가 지정한 사각형 영역을 바탕으로 GrabCut알고리즘을 사용하여 객체 추출               
  • 객체 추출 결과를 마스크 형태로 시각화               
  • 원본 이미지에서 배경을 제거하고 객체만 남은 이미지 출력               
  
#### 요구사항
  • cv.grabCut()를 사용하여 대화식 분할을 수행               
  • 초기 사각형 영역은 (x, y, width, height) 형식으로 설정               
  • 마스크를 사용하여 원본 이미지에서 배경을 제거               
  • matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화               
  
#### 힌트
  • cv.grabCut()에서 bgdModel과 fgdModel은 np.zeros((1, 65), np.float64)로 초기화               
  • 마스크 값은 cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD를 사용               
  • np.where()를 사용하여 마스크 값을 0 또는 1로 변경한 후 원본 이미지에 곱하여 배경을 제거               

### 코드
03.grabcut.py
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img = cv.imread('coffee cup.jpg')

if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 2. 마스크 생성
# 이미지 크기와 동일한 0으로 채워진 배열 생성
mask = np.zeros(img.shape[:2], np.uint8)

# 3. GrabCut에 사용할 배경/전경 모델 생성
# GrabCut 내부에서 사용하는 모델 (초기값 0)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 4. 초기 사각형 설정 (x, y, width, height)
# 이미지에 맞게 숫자는 조정 가능
rect = (50, 30, img.shape[1] - 100, img.shape[0] - 60)

# 5. GrabCut 수행
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# 6. 배경/전경 마스크 생성
mask2 = np.zeros(mask.shape, dtype=np.uint8)

# 배경으로 처리
mask2[mask == cv.GC_BGD] = 0
mask2[mask == cv.GC_PR_BGD] = 0

# 전경으로 처리
mask2[mask == cv.GC_FGD] = 1
mask2[mask == cv.GC_PR_FGD] = 1

# 7. 객체만 추출
result = img * mask2[:, :, np.newaxis]

# 8. BGR -> RGB 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

# 9. 마스크 시각화용 변환
mask_display = mask2 * 255    # 0/1 → 0/255

# 10. 결과 시각화
plt.figure(figsize=(15, 5))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# 마스크 이미지
plt.subplot(1, 3, 2)
plt.imshow(mask_display, cmap='gray')
plt.title('Mask Image')
plt.axis('off')

# 배경 제거 결과
plt.subplot(1, 3, 3)
plt.imshow(result_rgb)
plt.title('Foreground Extracted')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 핵심 코드

    • cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
      지정한 영역을 기준으로 전경과 배경을 분리하는 핵심 알고리즘
    
    • mask2[mask == cv.GC_FGD] = 1
      mask2[mask == cv.GC_PR_FGD] = 1
      전경(확실/가능)을 1로 설정하여 객체 영역을 구분하는 코드
    
    • result = img * mask2[:, :, np.newaxis]
      마스크를 적용하여 배경을 제거하고 객체만 남기는 코드

### 실행결과
<img width="2880" height="1124" alt="스크린샷 2026-03-19 135117" src="https://github.com/user-attachments/assets/46cafc28-51ca-4424-920d-a2e005b48b7f" />
