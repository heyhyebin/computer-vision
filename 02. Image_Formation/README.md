# 2주차

## 01. 체크보드 기반 카메라 캘리브레이션
### 문제
#### 설명
• 이미지에서 체크보드 코너를 검출하고 실제 좌표와 이미지 좌표의 대응관계를 이용하여 카메라 파라미터 추정
• 체크보드 패턴이 촬영된 여러 장의 이미지를 이용하여 카메라의 내부 행렬과 왜곡 계수를 계산하여 왜곡 보정
#### 요구사항
• 모든 이미지에서 체크보드 코너를 검출
• 체크보드의 실제 좌표와 이미지에서 찾은 코너 좌표를 구성
• cv2.calibrateCamera()를 사용하여 카메라 내부 행렬k와 왜곡 계수를 구함
• cv2.undistort()를 사용하여 왜곡 보정한 결과를 시각화
#### 힌트
• 체크보드 코너 검출은 cv2.findChessboardCorners() 사용
• 실제 좌표는 모든 이미지에서 동일한 격자 구조를 가짐(한칸의실제크기: 25mm)
• 체크보드는 평면 패턴이며 코너 검출에 실패한 이미지는 캘리브레이션에서 제외 가능

### 코드
01.Calibration.py
```python
import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
# z=0 평면 위에 일정한 간격의 형태로 배치
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = []
imgpoints = []

# 해당 이미지 파일 불러오기
images = glob.glob("calibration_images/left*.jpg")

# 이미지 크기 저장할 변수
img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    # 이미지 로드 실패 시
    if img is None:
        print("이미지 로드 실패:", fname)
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 코너 검출 그레이스 이미지에서 수행
    img_size = gray.shape[::-1]   # 이미지 크기 저장

    # 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    # 코너 검출 성공한 경우
    if ret:
        objpoints.append(objp)  # 실제 좌표

        # 코너 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # 시각화
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 실제 좌표(objpoints)와 이미지 좌표(imgpoints)를 이용해 카메라 내부 행렬 K, 왜곡 계수 dist를 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

print("Camera Matrix K:")   # 계산된 카메라 내부 행렬 출력
print(K)

print("\nDistortion Coefficients:")   # 계산된 왜곡 계수 출력
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
for fname in images:
    img = cv2.imread(fname)

    if img is None:
        continue

    h, w = img.shape[:2]
    # 왜곡 보정 후 사용할 최적의 새 카메라 행렬 계산
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    # 왜곡 보정 수행
    undistorted = cv2.undistort(img, K, dist, None, newK)
    # 검은 테두리 부분이 생길 수 있어 ROI만 잘라냄
    x, y, rw, rh = roi
    undistorted = undistorted[y:y+rh, x:x+rw]
    # 원본 / 왜곡 보정 결과 비교 출력
    cv2.imshow("Original", img)
    cv2.imshow("Undistorted", undistorted)
    # 아무 키나 누르면 다음 이미지로, ESC 누르면 종료
    key = cv2.waitKey()
    if key == 27:
        break

cv2.destroyAllWindows()

```

### 핵심 코드 설명

    • ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None) : 체크보드 패턴에서 내부 코너 위치를 검출
    
    • corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) : 검출된 코너 위치를 더 정확하게 보정
    
    • cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret) : 검출된 체크보드 코너를 이미지 위에 시각화
    
    • ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None) : 실제 좌표와 이미지 좌표를 이용해 카메라 내부 행렬(K)과 왜곡 계수(dist) 계산
    
    • newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h)) : 왜곡 보정 후 사용할 최적의 새 카메라 행렬 계산
    
    • undistorted = cv2.undistort(img, K, dist, None, newK) : 계산된 카메라 파라미터를 이용해 이미지 렌즈 왜곡 보정

### 실행결과
<img width="2531" height="1011" alt="image" src="https://github.com/user-attachments/assets/292e6918-e3fd-489e-a698-0f86c307cf4b" />
<img width="986" height="268" alt="image" src="https://github.com/user-attachments/assets/f8a7f5da-f9c0-4b2a-b25a-b94d7048eaaa" />



## 02. 이미지 Rotation & Transformation
### 문제
#### 설명
• 한 장의 이미지에 회전, 크기 조절, 평행이동을 적용
#### 요구사항
• 이미지의 중심 기준으로 +30도 회전
• 회전과 동시에 크기를 0.8로 조절
• 그 결과를 x축 방향으로 +80px, y축 방향으로 -40px만큼 평행이동
#### 힌트
• 회전 행렬은 cv2.getRotationMatrix2D()로 생성 가능
• 회전, 크기 조절, 평행이동은 cv2.warpAffine()로 적용
• 평행이동은 회전 행렬의 마지막 열 값을 조정하는 방식으로 반영

### 코드
02.AffineTransform.py
```python
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
```

### 핵심 코드 설명

    • h, w = img.shape[:2] : 입력 이미지의 높이와 너비 추출
    
    • center = (w // 2, h // 2) : 회전 기준이 될 이미지 중심 좌표 계산
    
    • M = cv2.getRotationMatrix2D(center, 30, 0.8) : 중심 기준 30도 회전과 0.8배 스케일을 위한 Affine 변환 행렬 생성
    
    • M[0, 2] += 80 : x축 방향으로 +80 픽셀 평행이동
    
    • M[1, 2] += -40 : y축 방향으로 -40 픽셀 평행이동
    
    • result = cv2.warpAffine(img, M, (w, h)) : 계산한 Affine 변환 행렬을 이용해 이미지 변환 수행



### 실행결과
<img width="2357" height="1578" alt="image" src="https://github.com/user-attachments/assets/7f43232d-28c3-45ef-92fc-b2b285884bb4" />



## 03. Stereo Disparity 기반 Depth 추정
### 문제
#### 설명
• 같은 장면을 왼쪽 카메라와 오른쪽 카메라에서 촬영한 두 장의 이미지를 이용해 깊이를 추정
• 두 이미지에서 같은 물체가 얼마나 옆으로 이동해 보이는지 계산하여 물체가 카메라에서 얼마나 떨어져 
있는지(depth)를 구할 수 있음
#### 요구사항
• 입력 이미지를 그레이스케일로 변환한 뒤 cv2.StereoBM_create()를 사용하여 disparity map 계산
• Disparity > 0인 픽셀만 사용하여 depth map 계산
• ROI Painting, Frog, Teddy 각각에 대해 평균 disparity와 평균 depth를 계산
• 세 ROI 중 어떤 영역이 가장 가까운지, 어떤 영역이 가장 먼지 해석
#### 힌트
• Disparity가 클 수록 물체는 더 가까움
• Depth는 𝑍=𝑓𝐵/𝑑로 계산 가능 
• Disparity map의 결과는 시각화하기 전에 정규화가 필요할 수 있음

### 코드
03.Depth.py
```python
import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("left.png")
right_color = cv2.imread("right.png")

# 둘 중 하나라도 없으면 에러 발생
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# 카메라 파라미터
# f: focal length, B: baseline(두 카메라 사이 거리)
f = 700.0
B = 0.12

# ROI 설정 (x, y, width, height)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# 블록 매칭 기반 StereoBM 객체 생성
stereo = cv2.StereoBM_create(numDisparities=16 * 6, blockSize=15)

# disparity 계산
# OpenCV는 disparity를 16배 스케일된 정수 형태로 주므로 16.0으로 나눔
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
depth_map = np.zeros_like(disparity, dtype=np.float32)      # disparity와 같은 크기의 depth 맵 생성
valid_mask = disparity > 0      # disparity가 0보다 큰 유효 픽셀만 사용
depth_map[valid_mask] = (f * B) / disparity[valid_mask]     # 깊이 계산

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # ROI 영역의 disparity / depth 추출
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    roi_valid = roi_disp > 0    # disparity가 0보다 큰 픽셀만 유효값으로 사용

    if np.any(roi_valid):
        # 유효 픽셀들의 평균 disparity / depth 계산
        mean_disp = np.mean(roi_disp[roi_valid])
        mean_depth = np.mean(roi_depth[roi_valid])
    else:
        # 유효 픽셀이 없으면 NaN 처리
        mean_disp = np.nan
        mean_depth = np.nan

    results[name] = {
        "mean_disparity": mean_disp,
        "mean_depth": mean_depth
    }

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("=== ROI별 평균 Disparity / Depth ===")
for name, values in results.items():
    print(f"{name}")
    print(f"  Mean Disparity: {values['mean_disparity']:.2f}")
    print(f"  Mean Depth    : {values['mean_depth']:.4f} m")
    print()

# 가장 가까운 / 가장 먼 ROI 찾기 / depth가 작을수록 가까움
valid_results = {
    name: values for name, values in results.items()
    if not np.isnan(values["mean_depth"])
}

if valid_results:
    nearest = min(valid_results.items(), key=lambda x: x[1]["mean_depth"])
    farthest = max(valid_results.items(), key=lambda x: x[1]["mean_depth"])

    print(f"가장 가까운 ROI: {nearest[0]} ({nearest[1]['mean_depth']:.4f} m)")
    print(f"가장 먼 ROI   : {farthest[0]} ({farthest[1]['mean_depth']:.4f} m)")


# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()
# disparity가 0 이하인 값은 무효값이므로 NaN 처리
disp_tmp[disp_tmp <= 0] = np.nan

# 전부 NaN이면 시각화 불가
if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

# 극단값 영향을 줄이기 위해 5~95 percentile 사용
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

# 0~1 범위로 정규화
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

# 0~255 범위의 8비트 이미지로 변환
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# 컬러맵 적용
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    # depth 역시 5~95 percentile 기준으로 정규화
    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6
    
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)    # 컬러맵 적용

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
# 원본 이미지를 복사해서 ROI 박스를 그릴 이미지 생성
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    # ROI 사각형 그리기
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # ROI 이름 텍스트 표시
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
# 결과 이미지 저장
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)
cv2.imwrite(str(output_dir / "right_roi.png"), right_vis)
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)

# ROI별 계산 결과를 텍스트 파일로 저장
with open(output_dir / "roi_results.txt", "w", encoding="utf-8") as f_out:
    f_out.write("=== ROI별 평균 Disparity / Depth ===\n")
    for name, values in results.items():
        f_out.write(f"{name}\n")
        f_out.write(f"  Mean Disparity: {values['mean_disparity']:.2f}\n")
        f_out.write(f"  Mean Depth    : {values['mean_depth']:.4f} m\n\n")

    if valid_results:
        f_out.write(f"가장 가까운 ROI: {nearest[0]} ({nearest[1]['mean_depth']:.4f} m)\n")
        f_out.write(f"가장 먼 ROI   : {farthest[0]} ({farthest[1]['mean_depth']:.4f} m)\n")


# -----------------------------
# 9. 출력
# -----------------------------
cv2.imshow("Left ROI", left_vis)
cv2.imshow("Right ROI", right_vis)
cv2.imshow("Disparity Color", disparity_color)
cv2.imshow("Depth Color", depth_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 핵심 코드 설명

    • left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY) : 왼쪽 컬러 이미지를 disparity 계산을 위해 그레이스케일 이미지로 변환
    
    • right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY) : 오른쪽 컬러 이미지를 그레이스케일 이미지로 변환
    
    • stereo = cv2.StereoBM_create(numDisparities=16 * 6, blockSize=15) : Stereo Block Matching 알고리즘 객체 생성
    
    • disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0 : 좌우 이미지의 disparity map 계산 (정수형 disparity 값을 실수형으로 변환 후 16으로 나눔)
    
    • depth_map[valid_mask] = (f * B) / disparity[valid_mask] : Depth 공식 𝑍=𝑓𝐵/𝑑를 이용해 실제 거리 계산
    
    • roi_disp = disparity[y:y+h, x:x+w] : ROI 영역의 disparity 값 추출
    
    • mean_disp = np.mean(roi_disp[roi_valid]) : ROI 내부 유효 disparity 평균 계산
    
    • mean_depth = np.mean(roi_depth[roi_valid]) : ROI 내부 유효 depth 평균 계산
    
    • disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET) : disparity map을 컬러맵으로 시각화
    
    • depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET) : depth map을 컬러맵으로 시각화

### 실행결과
<img width="889" height="744" alt="image" src="https://github.com/user-attachments/assets/a5b2682c-900c-44b1-9fb9-38b1c43646e6" />
