import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from tracker_utils import PathTracker
from heatmap import Heatmap
from loitering import LoiteringDetector

VIDEO_PATH = "input_video.mp4"

model = YOLO("yolov8m.pt")  # better accuracy than nano

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()
h, w, _ = frame.shape
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

tracker = DeepSort(max_age=30)
path_tracker = PathTracker()
heatmap = Heatmap(w, h)
loiter = LoiteringDetector(fps)

# ROI (example shelf area) — adjust manually
ROI = (int(w*0.6), int(h*0.3), int(w*0.95), int(h*0.9))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, iou=0.5)

    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if int(cls) == 0:  # person class
                x1, y1, x2, y2 = box
                w_box, h_box = x2 - x1, y2 - y1
                detections.append(([x1, y1, w_box, h_box], 1, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    active_ids = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        active_ids.append(track_id)

        path_tracker.update(track_id, (cx, cy))
        heatmap.update(cx, cy)

        # Check ROI
        x1_roi, y1_roi, x2_roi, y2_roi = ROI
        inside_roi = x1_roi < cx < x2_roi and y1_roi < cy < y2_roi

        loitering = loiter.update(track_id, inside_roi)

        cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (l, t-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if loitering:
            cv2.putText(frame, "LOITERING", (l, t-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw ROI
    cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (0,255,255), 2)

    # Current people count
    total_people = len(set(active_ids))
    cv2.putText(frame, f"Total = {total_people}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Retail Analytics", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

# Save heatmap
heatmap_img = heatmap.get_colored()
cv2.imwrite("heatmap.png", heatmap_img)

cv2.destroyAllWindows()
