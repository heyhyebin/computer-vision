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