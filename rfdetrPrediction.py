import cv2
import numpy as np
import warnings
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import cvzone

# suppress the upcoming torch.meshgrid warning
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument"
)

# --- Config ---
WEIGHTS_PATH = "checkpoint_best_regular.pth"
VIDEO_SOURCE = "Videos/warmupshots.mp4"
DISPLAY_W, DISPLAY_H = 800, 600
CAPTURE_W, CAPTURE_H = 1280, 720
CONF_THRESH = 0.3
TARGET_CLASS = 1  # the class_id you want to track 37 for coco base model
MIN_PTS_FIT = 5

# --- Init RF-DETR ---
model = RFDETRBase(pretrain_weights=WEIGHTS_PATH) #pretrain_weights=WEIGHTS_PATH
print(f"Loaded RF-DETR weights from {WEIGHTS_PATH}")

# --- Init ByteTrack ---
tracker = sv.ByteTrack()

# --- Buffers for trajectory ---
xs, ys = [], []
coeffs = None
paused = False

# --- Video capture & window setup ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

cv2.namedWindow("Trajectory Prediction", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trajectory Prediction", DISPLAY_W, DISPLAY_H)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

print("Press SPACE to pause/resume, 'q' to quit.")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Detect
        det = model.predict(frame[:, :, ::-1].copy(), threshold=CONF_THRESH)

        # 2) Filter class_id == TARGET_CLASS
        mask = (det.class_id == TARGET_CLASS)
        bboxes = det.xyxy[mask]
        confs = det.confidence[mask]

        if len(bboxes):
            # 3) Convert to Supervision Detections
            detections = sv.Detections(
                xyxy=bboxes,
                confidence=confs,
                class_id=det.class_id[mask]
            )

            # 4) Track with ByteTrack
            detections = tracker.update_with_detections(detections)

            if len(detections.xyxy) == 0:
                cv2.putText(frame, "Tracking lost", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # use first detected track
                x1, y1, x2, y2 = detections.xyxy[0]
                track_id = detections.tracker_id[0]

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # draw tracking box & label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID:{int(track_id)}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # 5) Bounce detection
                if ys and cy < ys[-1] - 2:
                    xs.clear()
                    ys.clear()
                    coeffs = None
                    print("Bounce → resetting trajectory fit")

                xs.append(cx)
                ys.append(cy)

                # draw past points
                for px, py in zip(xs, ys):
                    cv2.circle(frame, (px, py), 3, (0, 200, 0), -1)

                # 6) Fit parabola & draw prediction
                if len(xs) >= MIN_PTS_FIT:
                    coeffs = np.polyfit(xs, ys, 2)
                    a, b, c = coeffs
                    for t in range(0, CAPTURE_W, 5):
                        y_pred = int(a * t * t + b * t + c)
                        cv2.circle(frame, (t, y_pred), 2, (255, 0, 255), 3)

        # Show frame
        cvzone.putTextRect(frame,
                           "SPACE = Pause/Resume   |   Q = Quit",
                           (10, frame.shape[0] - 20),
                           # colorB=(255, 255, 255),
                           colorR=(0, 0, 0),
                           scale=3,
                           thickness=3
                           )
        cv2.imshow("Trajectory Prediction", frame)

    # Handle keypress
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # SPACE
        paused = not paused
        print("Paused" if paused else "Resumed")

cap.release()
cv2.destroyAllWindows()
