import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# -------------------- YOLO 설정 --------------------
CONF_THRES = 0.5
NMS_THRES = 0.4
INPUT_SIZE = 416

VIDEO_PATH = "slow_traffic_small.mp4"   # 필요하면 확장자 맞게 수정
CFG_PATH = "yolov3.cfg"                 # 탐색기에서 yolov3로 보여도 실제는 .cfg일 수 있음
WEIGHTS_PATH = "yolov3.weights"

# COCO 기준: person, bicycle, car, motorbike, bus, truck
TARGET_IDS = {0, 1, 2, 3, 5, 7}


# -------------------- SORT --------------------
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


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / (h + 1e-6)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    return np.array([
        x[0] - w / 2., x[1] - h / 2.,
        x[0] + w / 2., x[1] + h / 2.
    ], dtype=np.float32).reshape((1, 4))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ], dtype=np.float32)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ], dtype=np.float32)

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)[0]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)[0]


class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1

        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trks.append(pos)

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4] if len(dets) > 0 else np.empty((0, 4)),
            np.array(trks) if len(trks) > 0 else np.empty((0, 4)),
            self.iou_threshold
        )

        for d, t in matched:
            self.trackers[t].update(dets[d, :4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        ret = []
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


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


# -------------------- YOLO 검출 --------------------
def load_yolo():
    try:
        net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    except:
        net = cv2.dnn.readNetFromDarknet("yolov3", WEIGHTS_PATH)  # 확장자 숨김 대비

    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, out_layers


def detect(frame, net, out_layers):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes, confidences = [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            if conf < CONF_THRES or class_id not in TARGET_IDS:
                continue

            cx, cy = int(det[0] * w), int(det[1] * h)
            bw, bh = int(det[2] * w), int(det[3] * h)
            x = int(cx - bw / 2)
            y = int(cy - bh / 2)

            boxes.append([x, y, bw, bh])
            confidences.append(conf)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRES, NMS_THRES)

    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            dets.append([x1, y1, x2, y2, confidences[i]])

    if len(dets) == 0:
        return np.empty((0, 5))
    return np.array(dets, dtype=np.float32)


# -------------------- 실행 --------------------
def main():
    net, out_layers = load_yolo()
    tracker = Sort()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        print("VIDEO_PATH 확인:", VIDEO_PATH)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detect(frame, net, out_layers)
        tracks = tracker.update(dets)

        for trk in tracks:
            x1, y1, x2, y2, track_id = trk.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("SORT Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()