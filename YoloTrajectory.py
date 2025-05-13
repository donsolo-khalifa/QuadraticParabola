import math
import time
import numpy as np
import cv2
import cvzone
from ultralytics import YOLO

# for webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 1920)
# cap.set(4, 1080)

# for videos
cap = cv2.VideoCapture("Videos/shotComps.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Load YOLOv8 model ---
model = YOLO('randomNano.pt')
names = model.names
BALL_CLASS_ID = 0  # COCO index for sports ball 32 for yolov8 coco

# timing for FPS
prev_frame_time = 0
new_frame_time = 0

# trajectory storage
pos_x = []
pos_y = []
coff = None
min_points_for_fit = 5
bounce_thresh = 2  # pixels

paused = False


# target display size (so the window never exceeds this)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# create a resizable window and set its display size
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", DISPLAY_WIDTH, DISPLAY_HEIGHT)

while True:
    if not paused:

        new_frame_time = time.time()
        success, img = cap.read()
        if not success:
            break

        # run detection
        results = model(img, stream=True,verbose=False)

        # find the highest‐confidence sports ball this frame
        ball_center = None
        best_conf = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == BALL_CLASS_ID and conf > best_conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = x1 + (x2 - x1) // 2
                    cy = y1 + (y2 - y1) // 2
                    best_conf = conf
                    ball_center = (cx, cy)
                    # draw detection
                    cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorR=(255, 255, 0))
                    cvzone.putTextRect(img, f"ball {conf:.2f}", (x1, y1 - 10),
                                       colorR=(255, 255, 0), scale=1, thickness=1)

        # update trajectory
        if ball_center:
            cx, cy = ball_center

            # bounce detection: rising after falling
            if pos_y and (cy < pos_y[-1] - bounce_thresh):
                pos_x.clear()
                pos_y.clear()
                coff = None
                print("Bounce detected — restarting fit")

            pos_x.append(cx)
            pos_y.append(cy)

        # draw past trajectory
        for px, py in zip(pos_x, pos_y):
            cv2.circle(img, (px, py), 3, (0, 200, 0), cv2.FILLED)

        # fit & predict current arc
        if len(pos_x) >= min_points_for_fit:
            coff = np.polyfit(pos_x, pos_y, 2)
            for t in range(0, img.shape[1]):
                # for liner equation
                # y_pred = int(coff[0] * t + coff[1])

                y_pred = int(coff[0] * t * t + coff[1] * t + coff[2])
                cv2.circle(img, (t, y_pred), 2, (255, 0, 255), cv2.FILLED)

        # compute and show FPS
        fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
        prev_frame_time = new_frame_time
        cvzone.putTextRect(img, f'FPS {fps:.1f}', (10, 30),
                           colorR=(0, 200, 0), scale=1, thickness=2)

        # resize down to DISPLAY_* before showing
        cvzone.putTextRect(img,
                           "SPACE = Pause/Resume   |   Q = Quit",
                           (10, img.shape[0] - 20),
                           # colorB=(255, 255, 255),
                           colorR=(0, 0, 0),
                           scale=3,
                           thickness=3
                           )
        disp = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("Image", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # SPACE
        paused = not paused
        print("Paused" if paused else "Resumed")


cap.release()
cv2.destroyAllWindows()
