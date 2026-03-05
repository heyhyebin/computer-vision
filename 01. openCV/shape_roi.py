import cv2 as cv    # OpenCV 라이브러리를 cv라는 이름으로 불러오기
import sys  # 프로그램 종료를 위해 sys 모듈 사용

img = cv.imread('girl_laughing.jpg')    # 이미지 읽기
# 이미지 파일이 없을 경우
if img is None:
  sys.exit('파일이 존재하지 않습니다.')

clone = img.copy()  # 원본 이미지를 복사해서 clone에 저장 (원본 보존용)

x1, y1 = -1, -1 # 드래그할 마우스 초기 위치
roi = None

def select_roi(event, x, y, flags, param):  # 콜백 함수
    global x1, y1, img, clone, roi

    if event == cv.EVENT_LBUTTONDOWN:   # 왼쪽 마우스를 눌렸을 때
        x1, y1 = x, y   # 드래그 시작 좌표

    elif event == cv.EVENT_LBUTTONUP:   # 왼쪽 마우스 드래그 했을 때
        cv.rectangle(img, (x1,y1), (x,y), (0,255,0), 2) # 사각형 그리기

        # ROI 영역 추출 (numpy 슬라이싱)
        # y좌표 먼저, x좌표 
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

    elif key == ord('q'):   # 창 종료
        break

cv.destroyAllWindows()
