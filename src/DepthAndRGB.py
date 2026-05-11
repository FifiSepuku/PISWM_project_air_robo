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

    img = cv2.imread(os.path.join(rgb_path, img_name))

    # WAŻNE: nie IMREAD_UNCHANGED — SYNTHIA depth często i tak jest BGR encoded
    depth_img = cv2.imread(os.path.join(depth_path, img_name))



    if img is None or depth_img is None:
        continue

    h, w = img.shape[:2]
    frame = img.copy()

    results = model(img, classes=[0], conf=0.6)
    boxes = results[0].boxes

    for box in boxes:

        conf = float(box.conf[0])
        if conf < 0.75:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        roi_size = 30

        x_min = max(cx - roi_size, 0)
        x_max = min(cx + roi_size, w)
        y_min = max(cy - roi_size, 0)
        y_max = min(cy + roi_size, h)

        roi = depth_img[y_min:y_max, x_min:x_max]

        print("DEPTH RAW MIN/MAX:", depth_img.min(), depth_img.max())
        print("ROI RAW MIN/MAX:", roi.min(), roi.max())

        if roi.size == 0:
            continue

        roi = depth_img[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            continue

        # jeśli depth jest 2D (najczęściej case)
        if len(roi.shape) == 2:
            depth_values = roi.astype(np.float32)

        # jeśli 3 kanały (fallback)
        else:
            depth_values = roi[:, :, 0].astype(np.float32)

        # usuń zera
        depth_values = depth_values[depth_values > 0]

        if len(depth_values) == 0:
            continue

        distance_m = np.median(depth_values)
        print("DEPTH SAMPLE:", np.median(depth_values))
        # =========================
        # DRAW
        # =========================
        color = (0, 0, 255) if distance_m < 8 else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        cv2.putText(
            frame,
            f"{distance_m:.2f} m",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("SYNTHIA Depth Detection", frame)

    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()