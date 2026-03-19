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

# Canny 전에 블러 추가 (노이즈 제거)
gray = cv.GaussianBlur(gray, (5, 5), 0)

# 3. 캐니 에지 검출
# threshold1: 낮은 임계값, threshold2: 높은 임계값
edges = cv.Canny(gray, 100, 200)

# 4. 허프 변환으로 직선 검출
lines = cv.HoughLinesP(
    edges,                 # 입력 에지 이미지
    rho=1,                 # 거리 해상도 (픽셀 단위)
    theta=np.pi / 180,     # 각도 해상도 (라디안)
    threshold=150,         # 직선으로 인정할 최소 투표 수
    minLineLength=70,      # 최소 직선 길이
    maxLineGap=15          # 직선 사이 최대 간격
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