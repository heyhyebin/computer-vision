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

# 4. 특징점 검출 및 기술자(descriptor) 계산
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 5. FLANN 기반 매처 설정
# SIFT는 float descriptor이므로 KD-Tree 사용
index_params = dict(algorithm=1, trees=5)   # algorithm=1 : KDTree
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

# 6. 각 특징점마다 최근접 이웃 2개 찾기
matches = flann.knnMatch(des1, des2, k=2)

# 7. ratio test 적용
# 첫 번째 매칭 거리가 두 번째보다 충분히 작을 때만 좋은 매칭으로 인정
# 잘못된 매칭을 줄여 정확도를 높임
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 8. 좋은 매칭점만 시각화
matched_img = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS    # 매칭되지 않은 특징점은 그리지 않고, 연결된 매칭점만 표시
)

# 9. OpenCV는 BGR이므로 RGB로 변환
matched_img_rgb = cv.cvtColor(matched_img, cv.COLOR_BGR2RGB)

# 10. 결과 출력
plt.figure(figsize=(16, 8))
plt.imshow(matched_img_rgb)
plt.title(f'FLANN + KNN Match + Ratio Test ({len(good_matches)} good matches)')
plt.axis('off')
plt.show()