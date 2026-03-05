import cv2 as cv
import sys

img = cv.imread('girl_laughing.jpg')    # 이미지 읽기
# 이미지 파일이 없을 경우
if img is None:
  sys.exit('파일이 존재하지 않습니다.')

clone = img.copy()

x1, y1 = -1, -1
roi = None

def select_roi(event, x, y, flags, param):
    global x1, y1, img, clone, roi

    if event == cv.EVENT_LBUTTONDOWN:
        x1, y1 = x, y

    elif event == cv.EVENT_MOUSEMOVE and x1 != -1:
        img = clone.copy()
        cv.rectangle(img, (x1,y1), (x,y), (0,255,0), 2)

    elif event == cv.EVENT_LBUTTONUP:
        cv.rectangle(img, (x1,y1), (x,y), (0,255,0), 2)

        roi = clone[y1:y, x1:x]
        cv.imshow("ROI", roi)

cv.namedWindow("image")
cv.setMouseCallback("image", select_roi)

while True:

    cv.imshow("image", img)
    key = cv.waitKey(1) & 0xFF

    if key == ord('r'):     # r 입력한 경우 초기화
        img = clone.copy()

    elif key == ord('s') and roi is not None:   # s 입력한 경우 지정된 영역 저장
        cv.imwrite("roi.png", roi)

    elif key == ord('q'):
        break

cv.destroyAllWindows()