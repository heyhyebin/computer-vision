import cv2 as cv
import sys
import numpy as np

img = cv.imread('girl_laughing.jpg')  # 이미지 읽기
img_small=cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)   # 이미지 사이즈 반으로 축소

# 이미지 파일이 없을 경우
if img is None:
  sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일 변환
gray_small=cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5) # 이미지 사이즈 반으로 축소

gray_3 = cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR) # gray를 3채널로 변환 (hstack 맞추기 위해)

result = np.hstack((img_small, gray_3)) # 가로로 붙이기

# 출력
cv.imshow("Original vs Gray", result)
cv.waitKey()
cv.destroyAllWindows()