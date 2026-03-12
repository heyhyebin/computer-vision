import cv2

# 이미지 불러오기
img = cv2.imread("rose.png")

if img is None:
    raise FileNotFoundError("rose.png 파일을 찾을 수 없습니다.")

# 이미지 크기
h, w = img.shape[:2]

# 중심 좌표
center = (w // 2, h // 2)

# 1) 중심 기준 +30도 회전
# 2) 크기 0.8배 축소
M = cv2.getRotationMatrix2D(center, 30, 0.8)

# 3) x +80, y -40 평행이동
M[0, 2] += 80
M[1, 2] += -40

# Affine 변환 적용
result = cv2.warpAffine(img, M, (w, h))

# 결과 출력
cv2.imshow("Original", img)
cv2.imshow("Affine Transform Result", result)

cv2.waitKey()
cv2.destroyAllWindows()