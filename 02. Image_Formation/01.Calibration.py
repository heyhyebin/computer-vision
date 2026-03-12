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
