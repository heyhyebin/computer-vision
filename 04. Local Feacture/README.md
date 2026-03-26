# 4주차
## 01. SIFT를 이용한 특징점 검출 및 시각화
### 문제
#### 설명
  • 주어진 이미지(mot_color70.jpg)를 이용하여 SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용하여 특징점을
  검출하고 이를 시각화              

#### 요구사항
  • cv.SIFT_create()를 사용하여 SIFT 객체를 생성              
  • detectAndCompute()를 사용하여 특징점을 검출              
  • cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화              
  • matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력              

#### 힌트
  • SIFT_create()의 매개변수를 변경하며 특징점 검출 결과를 비교              
  • 특징점이 너무 많다면 nfeatures 값을 조정하여 제한              
  • cv.drawKeypoints()의 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS를 설정하면 특징점의
  방향과 크기도 표시              


### 코드
01_sift_keypoint.py
```python
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
```

### 핵심 코드
    • gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      SIFT 적용을 위해 이미지를 그레이스케일로 변환하는 코드
    
    • sift = cv.SIFT_create(nfeatures=300)
      SIFT 특징점 검출기를 생성하는 코드
    
    • keypoints, descriptors = sift.detectAndCompute(gray, None)
      특징점과 특징 벡터를 추출하는 핵심 코드
    
    • cv.drawKeypoints(...)
      특징점의 위치, 크기, 방향을 시각화하는 코드

### 실행 결과
<img width="2804" height="1324" alt="image" src="https://github.com/user-attachments/assets/eae80c24-927a-40f6-bbc5-e5558f2a36d2" />


## 02. SIFT를 이용한 두 영상 간 특징점 매
### 문제
#### 설명
  • 두 개의 이미지(mot_color70.jpg, mot_color80.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화     
  
#### 요구사항
  • cv.imread()를 사용하여 두 개의 이미지를 불러옴              
  • cv.SIFT_create()를 사용하여 특징점을 추출              
  • cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점을 매칭              
  • cv.drawMatches()를 사용하여 매칭 결과를 시각화              
  • matplotlib을 이용하여 매칭 결과를 출력              

#### 힌트
  • BFMatcher(cv.NORM_L2, crossCheck=True)를 사용하면 간단한 매칭이 가능              
  • FLANN 기반 매칭을 원하면 cv.FlannBasedMatcher()를 사용              
  • knnMatch()와 DMatch 객체를 활용하여 최근접 이웃 거리 비율을 적용하면 매칭 정확도를 높일 수 있음              

### 코드
02_sift_flann_matching.py
```python
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
```

### 핵심 코드
    • kp1, des1 = sift.detectAndCompute(gray1, None)
      두 이미지에서 특징점을 추출하는 코드
    
    • flann = cv.FlannBasedMatcher(index_params, search_params)
      FLANN 기반 매칭기를 생성하는 코드
    
    • matches = flann.knnMatch(des1, des2, k=2)
      각 특징점마다 가장 가까운 후보 2개를 찾는 코드
    
    • if m.distance < 0.7 * n.distance:
      ratio test를 통해 정확한 매칭만 선택하는 코드
    
    • cv.drawMatches(...)
      최종 매칭 결과를 시각화하는 코드

### 실행 결과
<img width="2488" height="1022" alt="image" src="https://github.com/user-attachments/assets/6f727d06-868b-40d0-ab06-9fe08d4663a3" />

## 03. 호모그래피를 이용한 이미지 정합 (Image Alignment)
### 문제
#### 설명
  • SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고, 이를 바탕으로 호모그래피를 계산하여 하나의 이미지 위에 정렬              
  • 샘플파일로 img1.jpg, imag2.jpg, imag3.jpg 중 2개를 선택              

#### 요구사항
  • cv.imread()를 사용하여 두 개의 이미지를 불러옴              
  • Cv.SIFT_create()를 사용하여 특징점을 검출              
  • cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭하고, 좋은 매칭점만 선별              
  • cv.findHomography()를 사용하여 호모그래피 행렬을 계산              
  • cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬              
  • 변환된 이미지(Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력              

#### 힌트
  • cv.findHomography()에서 cv.RANSAC을 사용하면 이상점(Outlier) 영향을 줄일 수 있음              
  • cv.warpPerspective()를 사용할 때 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정              
  • knnMatch()로 두 개의 최근접 이웃을 구한 뒤, 거리 비율이 임계값(예: 0.7) 미만인 매칭점만 선별              
### 코드
03_image_stitching.py
```python
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
```

### 핵심 코드
    • matches = bf.knnMatch(des2, des1, k=2)
      두 이미지 간 특징점 매칭 후보를 생성하는 코드
    
    • if m.distance < 0.7 * n.distance:
      ratio test로 신뢰도 높은 매칭만 선택하는 코드
    
    • src_pts / dst_pts 생성
      대응되는 특징점 좌표를 추출하는 코드
    
    • H, mask = cv.findHomography(...)
      RANSAC을 이용해 호모그래피(변환 행렬)를 계산하는 코드
    
    • cv.warpPerspective(...)
      계산된 변환 행렬을 이용해 이미지 정합을 수행하는 코드

### 실행 결과
<img width="2048" height="781" alt="image" src="https://github.com/user-attachments/assets/a4d66e58-efce-4f6f-a33b-251c56846830" />
