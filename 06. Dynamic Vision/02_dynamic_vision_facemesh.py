import cv2
import mediapipe as mp

# 1. 라이브러리 import 확인
print("1. import 성공")


# 2. Mediapipe FaceMesh 모듈 가져오기
mp_face_mesh = mp.solutions.face_mesh

# 3. FaceMesh 얼굴 랜드마크 검출기 생성
# static_image_mode=False : 실시간 영상 처리 모드
# max_num_faces=1 : 최대 1개의 얼굴만 검출
# refine_landmarks=True : 눈, 입술 등 세부 랜드마크 정교화
# min_detection_confidence : 얼굴 검출 최소 신뢰도
# min_tracking_confidence : 추적 최소 신뢰도
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("2. FaceMesh 생성 성공")


# 4. OpenCV로 웹캠 객체 생성
cap = cv2.VideoCapture(0)
print("3. 웹캠 객체 생성")


# 5. 웹캠이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("4. 웹캠 열기 성공")


# 6. 실시간 영상 프레임 반복 처리
while True:
    # 6-1. 웹캠에서 한 프레임 읽기
    ret, frame = cap.read()
    print("5. 프레임 읽기 시도")

    # 6-2. 프레임 읽기 실패 시 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 6-3. 좌우 반전하여 거울처럼 보이게 설정
    frame = cv2.flip(frame, 1)

    # 6-4. OpenCV의 BGR 영상을 Mediapipe용 RGB로 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 6-5. 얼굴 랜드마크 검출 수행
    results = face_mesh.process(rgb)

    # 6-6. 얼굴이 검출되면 랜드마크를 점으로 표시
    if results.multi_face_landmarks:
        h, w, _ = frame.shape

        # 검출된 각 얼굴에 대해 반복
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴의 468개 랜드마크 반복
            for lm in face_landmarks.landmark:
                # 정규화된 좌표를 실제 이미지 좌표로 변환
                x = int(lm.x * w)
                y = int(lm.y * h)

                # 랜드마크를 초록색 점으로 시각화
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # 7. 결과 영상 출력
    cv2.imshow("Face Mesh Landmarks", frame)

    # 8. ESC 키를 누르면 종료
    if cv2.waitKey(1) == 27:
        break


# 9. 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
