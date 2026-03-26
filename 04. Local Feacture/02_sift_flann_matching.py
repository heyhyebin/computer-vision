import cv2 as cv
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color80.jpg')

if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 2. 그레이스케일 변환
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 3. SIFT 객체 생성
sift = cv.SIFT_create()

# 4. 특징점 검출 및 descriptor 계산
# 두 이미지에서 특징점(keypoints)과 특징 벡터(descriptors) 추출
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 5. BFMatcher 생성 및 매칭
# crossCheck=True → 서로 가장 가까운 매칭만 인정하여 정확도 향상
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 6. 거리 기준으로 정렬
# 매칭 거리(distance)가 작은 순서대로 정렬 (좋은 매칭이 앞쪽)
matches = sorted(matches, key=lambda x: x.distance)

# 7. 상위 매칭점만 시각화
matched_img = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:50],   # 상위 50개만 표시
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS    # 매칭되지 않은 특징점은 제외하고, 연결된 매칭만 표시
)

# 8. BGR -> RGB 변환 후 출력
matched_img_rgb = cv.cvtColor(matched_img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(16, 8))
plt.imshow(matched_img_rgb)
plt.title(f'SIFT Feature Matching ({len(matches)} matches, showing top 50)')
plt.axis('off')
plt.show()
