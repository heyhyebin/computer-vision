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
