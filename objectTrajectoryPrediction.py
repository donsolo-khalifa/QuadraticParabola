import cv2
import numpy as np
import warnings
import supervision as sv
from rfdetr import RFDETRBase

# --- Config ---
WEIGHTS_PATH = "checkpoint_best_regular.pth"
VIDEO_SOURCE = "Videos/warmupshots.mp4"
OUTPUT_PATH = "tracked_output.mp4"
CAPTURE_W, CAPTURE_H = 1280, 720
CONF_THRESH = 0.3
TARGET_CLASS = 1  # your RF-DETR class_id to track
MIN_PTS_FIT = 5

# suppress the meshgrid warning
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument"
)

# --- Init detectors & trackers ---
model = RFDETRBase(pretrain_weights=WEIGHTS_PATH)
tracker = sv.ByteTrack()
box_annot = sv.BoxAnnotator()
label_annot = sv.LabelAnnotator()

# buffers for trajectory
xs, ys = [], []
coeffs = None


# --- Callback for process_video ---
def callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    global xs, ys, coeffs

    # 1) RF-DETR detection
    det = model.predict(frame[:, :, ::-1].copy(), threshold=CONF_THRESH)

    # 2) keep only your target class
    mask = det.class_id == TARGET_CLASS
    bboxes = det.xyxy[mask]
    confs = det.confidence[mask]
    clsids = det.class_id[mask]

    if len(bboxes):
        # 3) to supervision.Detections
        detections = sv.Detections(
            xyxy=bboxes,
            confidence=confs,
            class_id=clsids
        )

        # 4) ByteTrack update
        detections = tracker.update_with_detections(detections)

        if len(detections.xyxy):
            # unpack first tracked object
            x1, y1, x2, y2 = detections.xyxy[0]
            track_id = int(detections.tracker_id[0])

            # center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # bounce reset if direction flips up
            if ys and cy < ys[-1] - 2:
                xs.clear();
                ys.clear();
                coeffs = None

            xs.append(cx);
            ys.append(cy)

            # draw the fitted parabola once we have enough pts
            if len(xs) >= MIN_PTS_FIT:
                coeffs = np.polyfit(xs, ys, 2)
                a, b, c = coeffs
                for t in range(0, CAPTURE_W, 5):
                    y_pred = int(a * t * t + b * t + c)
                    cv2.circle(frame, (t, y_pred), 2, (255, 0, 255), 3)

        # draw past trajectory
        for px, py in zip(xs, ys):
            cv2.circle(frame, (px, py), 3, (0, 200, 0), -1)

        # 5) annotate boxes + IDs
        frame = box_annot.annotate(frame.copy(), detections=detections)
        # build ID labels
        id_labels = [f"ID:{int(i)}" for i in detections.tracker_id]
        frame = label_annot.annotate(frame, detections=detections, labels=id_labels)

    return frame


# --- Run and save ---
sv.process_video(
    source_path=VIDEO_SOURCE,
    target_path=OUTPUT_PATH,
    callback=callback
)

print(f"▶️ Saved annotated + tracked video to {OUTPUT_PATH}")
