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