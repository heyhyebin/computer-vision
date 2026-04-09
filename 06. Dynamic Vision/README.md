# 6주차
## 01. SORT 알고리즘을 활용한 다중 객체 추적기 구현
### 문제
#### 설명
* 이 실습에서는 SORT 알고리즘을 사용하여 비디오에서 다중 객체를 실시간으로 추적하는 프로그램을 구현합니다. 이
를 통해 객체 추적의 기본 개념과 SORT 알고리즘의 적용 방법을 학습할 수 있습니다.
#### 요구사항
* 객체 검출기 구현: YOLOv3와 같은 사전 훈련된 객체 검출 모델을 사용하여 각 프레임에서 객체를 검출합니다.
* mathworks.comSORT 추적기 초기화: 검출된 객체의 경계 상자를 입력으로 받아 SORT 추적기를 초기화합니다.
* 객체 추적: 각 프레임마다 검출된 객체와 기존 추적 객체를 연관시켜 추적을 유지합니다.
* 결과 시각화: 추적된 각 객체에 고유 ID를 부여하고, 해당 ID와 경계 상자를 비디오 프레임에 표시하여 실시간으로 출
력합니다.
#### 힌트
* 객체 검출: OpenCV의 DNN 모듈을 사용하여 YOLOv3 모델을 로드하고, 각 프레임에서 객체를 검출할 수 있습니다.
* SORT 알고리즘: SORT 알고리즘은 칼만 필터와 헝가리안 알고리즘을 사용하여 객체의 상태를 예측하고, 데이터 연
관을 수행합니다.
* 추적 성능 향상: 객체의 appearance 정보를 활용하는 Deep SORT와 같은 확장된 알고리즘을 사용하면 추적 성능을
향상시킬 수 있습니다.


### 코드
01_dynamic_vision_sort_tracking.py
```python
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# 1. YOLO 설정
# 객체 검출에 필요한 임계값, 입력 크기, 파일 경로 설정
CONF_THRES = 0.5       # 객체로 판단할 최소 신뢰도
NMS_THRES = 0.4        # 중복 박스 제거 기준
INPUT_SIZE = 416       # YOLO 입력 이미지 크기

VIDEO_PATH = "slow_traffic_small.mp4"   # 비디오 파일 경로
CFG_PATH = "yolov3.cfg"                 # YOLO 구조 파일
WEIGHTS_PATH = "yolov3.weights"         # YOLO 학습 가중치 파일

# COCO 클래스 중 교통 관련 객체만 추적
# person(0), bicycle(1), car(2), motorbike(3), bus(5), truck(7)
TARGET_IDS = {0, 1, 2, 3, 5, 7}


# 2. IoU 계산 함수
# 두 bounding box의 겹치는 비율을 계산
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    return wh / (
        (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
        (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh + 1e-6
    )


# 3. bounding box를 Kalman Filter 입력 형식으로 변환
# [x1, y1, x2, y2] -> [중심 x, 중심 y, 면적 s, 종횡비 r]
def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / (h + 1e-6)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


# 4. Kalman Filter 상태를 bounding box로 변환
# [중심 x, 중심 y, 면적 s, 종횡비 r] -> [x1, y1, x2, y2]
def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    return np.array([
        x[0] - w / 2., x[1] - h / 2.,
        x[0] + w / 2., x[1] + h / 2.
    ], dtype=np.float32).reshape((1, 4))


# 5. 개별 객체를 추적하는 Kalman Filter 클래스
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # 상태 벡터 7개, 관측 벡터 4개로 Kalman Filter 생성
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 상태 전이 행렬
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ], dtype=np.float32)

        # 측정 행렬
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ], dtype=np.float32)

        # 오차 공분산 설정
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[4:, 4:] *= 0.01

        # 초기 bounding box 상태 설정
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # 객체 고유 ID 부여
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # 추적 상태 변수
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    # 5-1. 새로운 관측값으로 tracker 상태 업데이트
    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    # 5-2. 다음 프레임 위치 예측
    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)[0]

    # 5-3. 현재 추적 상태 반환
    def get_state(self):
        return convert_x_to_bbox(self.kf.x)[0]


# 6. SORT 메인 클래스
# 여러 객체를 한 번에 추적하는 다중 객체 추적기
class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age              # 오래 갱신 안 된 객체 제거 기준
        self.min_hits = min_hits            # 최소 추적 횟수
        self.iou_threshold = iou_threshold  # IoU 매칭 기준
        self.trackers = []
        self.frame_count = 0

    # 6-1. 매 프레임마다 tracker를 업데이트
    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1

        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()  # 기존 tracker 위치 예측
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trks.append(pos)

        for t in reversed(to_del):
            self.trackers.pop(t)

        # 검출 결과와 기존 tracker 매칭
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4] if len(dets) > 0 else np.empty((0, 4)),
            np.array(trks) if len(trks) > 0 else np.empty((0, 4)),
            self.iou_threshold
        )

        # 매칭된 tracker는 새 검출값으로 업데이트
        for d, t in matched:
            self.trackers[t].update(dets[d, :4])

        # 새로 검출된 객체는 tracker 추가
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        ret = []
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]
            d = trk.get_state()

            # 현재 프레임에서 유효한 tracker만 반환
            if (trk.time_since_update < 1):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


# 7. 검출 결과와 tracker를 IoU 기준으로 매칭
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(row_ind, col_ind))) if len(row_ind) > 0 else np.empty((0, 2), dtype=int)

    unmatched_dets = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0] if len(matched_indices) > 0 else True:
            unmatched_dets.append(d)

    unmatched_trks = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1] if len(matched_indices) > 0 else True:
            unmatched_trks.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


# 8. YOLO 모델 로드
# cfg 파일과 weights 파일을 읽어 네트워크 생성
def load_yolo():
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, out_layers


# 9. 한 프레임에서 객체 검출
# YOLO를 사용해 관심 객체의 bounding box 추출
def detect(frame, net, out_layers):
    h, w = frame.shape[:2]

    # 9-1. 이미지 전처리
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
    net.setInput(blob)

    # 9-2. YOLO 추론
    outputs = net.forward(out_layers)

    boxes, confidences = [], []

    # 9-3. 검출 결과 처리
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            # 신뢰도와 관심 클래스 조건 확인
            if conf < CONF_THRES or class_id not in TARGET_IDS:
                continue

            cx, cy = int(det[0] * w), int(det[1] * h)
            bw, bh = int(det[2] * w), int(det[3] * h)
            x = int(cx - bw / 2)
            y = int(cy - bh / 2)

            boxes.append([x, y, bw, bh])
            confidences.append(conf)

    # 9-4. NMS로 중복 제거
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRES, NMS_THRES)

    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            dets.append([x, y, x + bw, y + bh, confidences[i]])

    return np.array(dets) if len(dets) > 0 else np.empty((0, 5))


# 10. 메인 실행 함수
# 비디오를 읽으면서 객체 검출과 추적을 반복 수행
def main():
    net, out_layers = load_yolo()   # YOLO 모델 로드
    tracker = Sort()                # SORT 추적기 생성

    cap = cv2.VideoCapture(VIDEO_PATH)  # 비디오 열기

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 10-1. 객체 검출
        dets = detect(frame, net, out_layers)

        # 10-2. 객체 추적
        tracks = tracker.update(dets)

        # 10-3. 결과 시각화
        for trk in tracks:
            x1, y1, x2, y2, track_id = trk.astype(int)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("SORT Tracking", frame)

        # 10-4. ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# 11. 프로그램 시작점
if __name__ == "__main__":
    main()
```

### 핵심코드
    • net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    YOLOv3 모델을 불러오는 코드
    
    • blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
    현재 프레임을 YOLO 입력 형식으로 전처리하는 코드
    
    • outputs = net.forward(out_layers)
    YOLO를 이용해 객체를 검출하는 핵심 코드
    
    • tracker = Sort()
    SORT 다중 객체 추적기를 초기화하는 코드
    
    • tracks = tracker.update(dets)
    현재 프레임의 검출 결과와 이전 추적 결과를 연결해 같은 객체의 ID를 유지하는 코드
    
    • cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    검출·추적된 객체의 위치를 경계 상자로 표시하는 코드
    
    • cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    객체마다 고유 ID를 화면에 출력하는 코드

### 실행결과
<img width="642" height="392" alt="image" src="https://github.com/user-attachments/assets/9bf8119b-8a2c-404a-aadb-7e01b179c0d4" />




--------------------------------------------------------




## 02. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화
### 문제
#### 설명
* Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크를 추출하고, 이를 실시간 영상에 시각화하는 프로그
램을 구현합니다.
#### 요구사항
* Mediapipe의 FaceMesh 모듈을 사용하여 얼굴 랜드마크 검출기를 초기화합니다.
* OpenCV를 사용하여 웹캠으로부터 실시간 영상을 캡처합니다.
* 검출된 얼굴 랜드마크를 실시간 영상에 점으로 표시합니다.
* ESC 키를 누르면 프로그램이 종료되도록 설정합니다.
#### 힌트
* Mediapipe의 solutions.face_mesh를 사용하여 얼굴 랜드마크 검출기를 생성할 수 있습니다.
* 검출된 랜드마크 좌표를 이용하여 OpenCV의 circle 함수를 사용해 각 랜드마크를 시각화할 수 있습니다.
* 랜드마크 좌표는 정규화되어 있으므로, 이미지 크기에 맞게 변환이 필요합니다.

### 코드
02_dynamic_vision_facemesh.py
```python
import cv2
import mediapipe as mp

# 1. 라이브러리 import 확인
print("1. import 성공")


# 2. Mediapipe FaceMesh 모듈 가져오기
mp_face_mesh = mp.solutions.face_mesh

# 3. FaceMesh 얼굴 랜드마크 검출기 생성
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,         # 실시간 영상 처리 모드
    max_num_faces=1,                 # 최대 1개의 얼굴만 검출
    refine_landmarks=True,           # 눈, 입술 등 세부 랜드마크 정교화
    min_detection_confidence=0.5,    # 얼굴 검출 최소 신뢰도
    min_tracking_confidence=0.5      # 추적 최소 신뢰도
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
```

### 핵심코드
    • mp_face_mesh = mp.solutions.face_mesh
    Mediapipe의 FaceMesh 모듈을 불러오는 코드
    
    • face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    얼굴 랜드마크 검출기를 생성하는 코드
    
    • cap = cv2.VideoCapture(0)
    웹캠으로부터 실시간 영상을 입력받는 코드
    
    • results = face_mesh.process(rgb)
    현재 프레임에서 얼굴 랜드마크를 검출하는 핵심 코드
    
    • x = int(lm.x * w), y = int(lm.y * h)
    정규화된 랜드마크 좌표를 실제 이미지 좌표로 변환하는 코드
    
    • cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    검출된 랜드마크를 영상 위에 점으로 표시하는 코드
    
    • if cv2.waitKey(1) == 27:
    ESC 키를 눌렀을 때 프로그램을 종료하는 코드

### 실행결과
<img width="684" height="914" alt="image" src="https://github.com/user-attachments/assets/3e74d4a6-5d61-4e51-8eff-a6e92b69eb6f" />





