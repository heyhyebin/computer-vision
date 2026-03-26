import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img1 = cv.imread('img1.jpg')   # 기준 이미지
img2 = cv.imread('img2.jpg')   # 변환할 이미지

if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 2. 그레이스케일 변환
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 3. SIFT 특징점 검출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 4. BFMatcher + knnMatch
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des2, des1, k=2)
# des2 -> des1 로 맞춰서 img2를 img1 쪽으로 warp

# 5. 좋은 매칭점 선별 (ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

if len(good_matches) < 4:
    print("호모그래피를 계산하기 위한 좋은 매칭점이 부족합니다.")
    exit()

# 6. 대응점 추출
# src_pts: 변환할 이미지(img2)의 좌표
# dst_pts: 기준 이미지(img1)의 대응 좌표
src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 7. 호모그래피 계산 (RANSAC 사용)
# RANSAC을 사용해 이상치(outlier)를 제외하며 호모그래피 행렬 계산
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

if H is None:
    print("호모그래피 계산에 실패했습니다.")
    exit()

# 8. 이미지 정합
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 계산된 호모그래피를 이용해 img2를 img1 좌표계에 맞게 변환
warped = cv.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))

# 기준 이미지(img1)를 왼쪽에 배치
warped[0:h1, 0:w1] = img1

# 9. 매칭 결과 시각화
matching_result = cv.drawMatches(
    img2, kp2,
    img1, kp1,
    good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 10. BGR -> RGB 변환
warped_rgb = cv.cvtColor(warped, cv.COLOR_BGR2RGB)
matching_rgb = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)

# 11. 결과 출력
plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)
plt.imshow(matching_rgb)
plt.title(f'Matching Result ({len(good_matches)} good matches)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(warped_rgb)
plt.title('Warped Image / Image Alignment')
plt.axis('off')

plt.tight_layout()
plt.show()