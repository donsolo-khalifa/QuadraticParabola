import cv2
import numpy as np
import math
import cvzone

# --- Configuration ---
VIDEO_SOURCE = 'Videos/escapeball.mp4'  # or 0 for webcam
CAPTURE_WIDTH = 1280  # desired capture width
CAPTURE_HEIGHT = 720  # desired capture height
WINDOW_NAME = "Trajectory Prediction"
WINDOW_WIDTH = 800  # display window width
WINDOW_HEIGHT = 600  # display window height
# Choose tracker: 'MOSSE', 'CSRT', 'KCF', 'TLD', 'MedianFlow', 'MIL', 'Boosting'
TRACKER_TYPE = 'CSRT'






# --- Helper: create tracker instance ---
def create_tracker(name):
    Tracker = cv2.legacy
    if name == 'MOSSE':
        return Tracker.TrackerMOSSE_create()
    elif name == 'CSRT':
        return Tracker.TrackerCSRT_create()
    elif name == 'KCF':
        return Tracker.TrackerKCF_create()
    elif name == 'TLD':
        return Tracker.TrackerTLD_create()
    elif name == 'MedianFlow':
        return Tracker.TrackerMedianFlow_create()
    elif name == 'MIL':
        return Tracker.TrackerMIL_create()
    elif name == 'Boosting':
        return Tracker.TrackerBoosting_create()
    else:
        raise ValueError(f"Unknown tracker type '{name}'")


# --- Initialize video capture ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

# --- Frame selection loop ---
cv2.namedWindow("Frame Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame Preview", WINDOW_WIDTH, WINDOW_HEIGHT)
print("Press SPACE for next frame, ENTER to select frame for ROI, 'q' to quit.")
frame_for_roi = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames.")
        break
    cvzone.putTextRect(
        frame,
        "Space = Next frame, Enter = Select frame, q = exit",
        (10, frame.shape[0] - 20),
        colorB=(255, 255, 255),
        colorR=(0, 0, 0),
        scale=2.3,
        thickness=2
    )
    cv2.imshow("Frame Preview", frame)


    key = cv2.waitKey(0) & 0xFF
    if key == 32:  # SPACE → next frame
        continue
    elif key == 13:  # ENTER → select this frame
        frame_for_roi = frame.copy()
        cvzone.putTextRect(
            frame_for_roi,
            "Drag to select ROI with mouse, then press ENTER",
            (10, frame_for_roi.shape[0] - 20),
            colorB=(255, 255, 255),
            colorR=(0, 0, 0),
            scale=2.5,
            thickness=2
        )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Frame Preview")
if frame_for_roi is None:
    print("Frame not selected. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# --- ROI selection & tracker init ---
bbox = cv2.selectROI(WINDOW_NAME, frame_for_roi, False)
cv2.destroyWindow(WINDOW_NAME)
tracker = create_tracker(TRACKER_TYPE)
tracker.init(frame_for_roi, bbox)

# --- Trajectory & bounce variables ---
pos_list_x = []
pos_list_y = []
coff = None
min_points_for_fit = 5
paused = False  # flag for pause state

# --- Prepare display window ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

# --- Main tracking loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox = tracker.update(frame)
    if not success:
        cv2.putText(frame, "Tracking lost. Press 'r' to re-select.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        x, y, w, h = map(int, bbox)
        cx, cy = x + w // 2, y + h // 2

        # draw box & center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

        # --- Bounce detection ---
        if pos_list_y and (cy < pos_list_y[-1] - 2):
            # ball is now moving up after falling → bounce
            pos_list_x.clear()
            pos_list_y.clear()
            coff = None
            print("Bounce detected — restarting fit")

        # store trajectory point
        pos_list_x.append(cx)
        pos_list_y.append(cy)

        # draw past trajectory
        for px, py in zip(pos_list_x, pos_list_y):
            cv2.circle(frame, (px, py), 3, (0, 200, 0), cv2.FILLED)

        # fit & predict parabola
        if len(pos_list_x) >= min_points_for_fit:
            coff = np.polyfit(pos_list_x, pos_list_y, 2)
            for t in range(0, CAPTURE_WIDTH, 3):
                y_pred = int(coff[0] * t ** 2 + coff[1] * t + coff[2])
                cv2.circle(frame, (t, y_pred), 2, (255, 0, 255), cv2.FILLED)

    cvzone.putTextRect(
        frame,
        "SPACE = Pause/Resume | R = Re-select ROI | Q = Quit",
        (10, frame.shape[0] - 20),
        colorB=(255, 255, 255),
        colorR=(0, 0, 0),
        scale=2.3,
        thickness=2
    )

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        # reset everything and re-select ROI
        frame_for_roi = frame.copy()

        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pos_list_x.clear()
        pos_list_y.clear()
        coff = None

        print("Press SPACE for next frame, ENTER to select frame, 'q' to quit.")
        # frame_for_roi = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.namedWindow("Frame Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame Preview", WINDOW_WIDTH, WINDOW_HEIGHT)

            cvzone.putTextRect(
                frame,
                "SPACE = Next Frame | Enter = Select Frame | Q = Quit",
                (0, frame.shape[0] - 20),
                colorB=(255, 255, 255),
                colorR=(0, 0, 0),
                scale=2.3,
                thickness=2
            )

            cv2.imshow("Frame Preview", frame)
            k = cv2.waitKey(0) & 0xFF
            if k == 32:
                continue
            elif k == 13:
                frame_for_roi = frame.copy()
                cvzone.putTextRect(
                    frame_for_roi,
                    "Drag to select ROI with mouse, then press ENTER",
                    (10, frame_for_roi.shape[0] - 20),
                    colorB=(255, 255, 255),
                    colorR=(0, 0, 0),
                    scale=2.5,
                    thickness=2
                )
                break
            elif k == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
        cv2.destroyWindow("Frame Preview")
        if frame_for_roi is None:
            break

        bbox = cv2.selectROI(WINDOW_NAME, frame_for_roi, False)
        tracker = create_tracker(TRACKER_TYPE)
        tracker.init(frame_for_roi, bbox)

    elif key == 32:
        # pause/resume on SPACE
        paused = True
        print("Paused. Press SPACE to resume.")

        while paused:
            k2 = cv2.waitKey(0) & 0xFF
            if k2 == 32:
                paused = False
                print("Resumed.")

cap.release()
cv2.destroyAllWindows()
