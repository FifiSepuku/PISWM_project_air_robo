import cv2
import os
import json
import numpy as np
from ultralytics import YOLO

# load config
with open("../config/config.json", "r") as f:
    config = json.load(f)

# model
model = YOLO(config["model_path"])

# paths
rgb_path = config["left_path"]
depth_path = config["depth_path"]

# process dataset
for img_name in os.listdir(rgb_path):

    if not img_name.endswith(".png"):
        continue

    img = cv2.imread(os.path.join(rgb_path, img_name))
    depth_img = cv2.imread(os.path.join(depth_path, img_name))

    if img is None or depth_img is None:
        continue

    h, w = img.shape[:2]
    frame = img.copy()

    # detect people
    results = model(img, classes=[0], conf=config["confidence_threshold"])
    boxes = results[0].boxes

    for box in boxes:

        conf = float(box.conf[0])
        if conf < config["confidence_threshold"]:
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

        # debug depth range
        print("DEPTH RAW MIN/MAX:", depth_img.min(), depth_img.max())
        print("ROI RAW MIN/MAX:", roi.min(), roi.max())

        if roi.size == 0:
            continue

        # depth extraction
        if len(roi.shape) == 2:
            depth_values = roi.astype(np.float32)
        else:
            depth_values = roi[:, :, 0].astype(np.float32)

        depth_values = depth_values[depth_values > 0]

        if len(depth_values) == 0:
            continue

        distance_m = np.median(depth_values)

        print("DEPTH SAMPLE:", distance_m)

        # draw results
        color = (0, 0, 255) if distance_m < config["collision_threshold"] else (0, 255, 0)

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