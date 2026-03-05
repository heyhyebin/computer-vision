import cv2 as cv    # OpenCV 라이브러리를 cv라는 이름으로 불러오기
import sys          # 프로그램 종료를 위해 sys 모듈 사용

img = cv.imread('girl_laughing.jpg')    # 이미지 읽기
if img is None:                         # 이미지 파일이 없을 경우
    sys.exit('파일이 존재하지 않습니다.')

clone = img.copy()  # 원본 이미지를 복사해서 clone에 저장 (원본 보존용)

drawing = False
x1, y1 = -1, -1     # 드래그 시작 좌표
roi = None          # 선택된 ROI 저장용

def select_roi(event, x, y, flags, param):
    global x1, y1, drawing, img, clone, roi

    if event == cv.EVENT_LBUTTONDOWN:   # 마우스 왼쪽 버튼 누름: 드래그 시작
        drawing = True
        x1, y1 = x, y

    elif event == cv.EVENT_MOUSEMOVE and drawing:
        # 2) 드래그 중(마우스 이동): 사각형이 실시간으로 보이도록 계속 갱신
        img = clone.copy()  # ★ 매번 원본으로 되돌린 뒤
        cv.rectangle(img, (x1, y1), (x, y), (0, 255, 0), 2)  # ★ 현재 마우스 위치까지 임시 사각형

    elif event == cv.EVENT_LBUTTONUP:   # 마우스 버튼 뗌: 최종 ROI 확정
        drawing = False

        # 드래그 방향이 반대여도 ROI가 정상 나오도록 좌표 정렬
        x_min, x_max = min(x1, x), max(x1, x)
        y_min, y_max = min(y1, y), max(y1, y)

        img = clone.copy()  # 최종 사각형도 깔끔하게 표시되도록 원본에서 다시 그림
        cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # ROI 영역 추출 (numpy 슬라이싱: [y범위, x범위])
        roi = clone[y_min:y_max, x_min:x_max]

        # ROI가 비어있지 않을 때만 띄우기
        if roi.size > 0:
            cv.imshow("ROI", roi)

cv.namedWindow("image")
cv.setMouseCallback("image", select_roi)

while True:
    cv.imshow("image", img)
    key = cv.waitKey(1) & 0xFF

    if key == ord('r'):     # r 입력한 경우 초기화
        img = clone.copy()
        roi = None          # ROI도 같이 초기화

    elif key == ord('s') and roi is not None:   # s 입력한 경우 지정된 영역 저장
        cv.imwrite("roi.png", roi)

    elif key == ord('q'):   # q 입력 시 종료
        break

cv.destroyAllWindows()
