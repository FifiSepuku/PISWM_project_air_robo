import cv2
import os
import numpy as np
from ultralytics import YOLO

# =========================
# MODEL
# =========================

model = YOLO("yolov8s.pt")

# =========================
# PATHS
# =========================

rgb_path = r"D:\SYNTHIA-SEQS-04-FALL\RGB\Stereo_Left\Omni_F"
depth_path = r"D:\SYNTHIA-SEQS-04-FALL\Depth\Stereo_Left\Omni_F"

# =========================
# MAIN LOOP
# =========================

for img_name in os.listdir(rgb_path):

    if not img_name.endswith(".png"):
        continue

    # =========================
    # LOAD RGB
    # =========================

    img = cv2.imread(os.path.join(rgb_path, img_name))
    if img is None:
        continue

    # =========================
    # LOAD DEPTH
    # =========================

    depth_img = cv2.imread(os.path.join(depth_path, img_name))
    if depth_img is None:
        continue

    h, w = img.shape[:2]
    frame = img.copy()

    # =========================
    # YOLO
    # =========================

    results = model(img, classes=[0], conf=0.3)
    boxes = results[0].boxes

    # =========================
    # PROCESS BOXES
    # =========================

    for box in boxes:

        conf = float(box.conf[0])
        if conf < 0.3:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # =========================
        # DEPTH ROI
        # =========================

        roi_size = 10

        x_min = max(center_x - roi_size, 0)
        x_max = min(center_x + roi_size, w - 1)
        y_min = max(center_y - roi_size, 0)
        y_max = min(center_y + roi_size, h - 1)

        roi = depth_img[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            continue

        B = roi[:, :, 0].astype(np.int32)
        G = roi[:, :, 1].astype(np.int32)
        R = roi[:, :, 2].astype(np.int32)

        depth_values = (
            5000 *
            (R + G * 256 + B * 256 * 256)
            / (256 * 256 * 256 - 1)
        )

        depth_values = depth_values[depth_values > 0]

        if len(depth_values) == 0:
            continue

        distance = float(np.median(depth_values))
        distance = round(distance, 2)

        # =========================
        # DRAW BOX
        # =========================

        color = (0, 0, 255) if distance < 8 else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # center point
        cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

        # distance label
        cv2.putText(
            frame,
            f"{distance:.2f} m",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # =========================
    # SHOW
    # =========================

    cv2.imshow("Pedestrian Distance", frame)

    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()