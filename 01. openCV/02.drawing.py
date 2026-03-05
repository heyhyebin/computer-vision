import cv2 as cv    # OpenCV 라이브러리를 cv라는 이름으로 불러오기
import sys  # 프로그램 종료를 위해 sys 모듈 사용
import numpy as np

img = cv.imread('girl_laughing.jpg')    # 이미지 읽기
# 이미지 파일이 없을 경우
if img is None:
  sys.exit('파일이 존재하지 않습니다.')

brush_size = 5  # 초기 붓 크기 지정

def draw(event, x, y, flags, param):  # 콜백 함수
    global brush_size
    # 좌클릭하여 마우스를 움직이는 경우 원으로 파란색으로 그리기
    if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON):
        cv.circle(img, (x,y), brush_size, (255,0,0), -1)
    # 우클릭하여 마우스를 움직이는 경우 원으로 빨간색으로 그리기
    if event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON):
        cv.circle(img, (x,y), brush_size, (0,0,255), -1)

cv.namedWindow("Paint")
cv.setMouseCallback("Paint", draw)

while True:
    cv.imshow("Paint", img)
    key = cv.waitKey(1) & 0xFF  # 키 입력 확인

    if key == ord('+'):   # + 입력한 경우 붓 크기 증가
        brush_size = min(15, brush_size+1)

    elif key == ord('-'): # - 입력한 경우 붓 크기 감소
        brush_size = max(1, brush_size-1)

    elif key == ord('q'): # 창 종료
        break

cv.destroyAllWindows()
