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