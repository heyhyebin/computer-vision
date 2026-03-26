import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img = cv.imread('mot_color70.jpg')

if img is None:
    print("이미지를 불러올 수 없습니다: mot_color70.jpg")
    exit()

# OpenCV는 BGR로 읽기 때문에 grayscale 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성
# 특징점이 너무 많으면 nfeatures 값을 더 줄이면 됨
sift = cv.SIFT_create(nfeatures=300)

# 3. 특징점 검출 및 descriptor 계산
# keypoints: 특징점 위치 정보, descriptors: 특징점의 특징 벡터
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 4. 특징점 시각화
img_keypoints = cv.drawKeypoints(
    img,
    keypoints,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS     # 특징점의 위치뿐 아니라 크기와 방향 정보까지 함께 표시
)

# 5. matplotlib 출력용 BGR -> RGB 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_keypoints_rgb = cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB)

# 6. 결과 출력
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_keypoints_rgb)
plt.title(f'SIFT Keypoints ({len(keypoints)} points)')
plt.axis('off')

plt.tight_layout()
plt.show()