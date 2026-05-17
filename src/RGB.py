import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
with open("../config/config.json", "r") as f:
    config = json.load(f)

# camera parameters
FOCAL_LENGTH = config["focal_length"]
BASELINE = config["baseline"]

# YOLO model
model = YOLO(config["model_path"])

# image paths
left_path = config["left_path"]
right_path = config["right_path"]

# stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5
)

# process frames
for name in os.listdir(left_path):

    if not name.endswith(".png"):
        continue

    left = cv2.imread(os.path.join(left_path, name))
    right = cv2.imread(os.path.join(right_path, name))

    if left is None or right is None:
        continue

    h, w = left.shape[:2]
    frame = left.copy()

    # grayscale conversion
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # disparity map
    disparity = stereo.compute(
        gray_l,
        gray_r
    ).astype(np.float32) / 16.0

    # pedestrian detection
    results = model(left, classes=[0], conf=config["confidence_threshold"])
    boxes = results[0].boxes

    # analyze detections
    for box in boxes:

        conf = float(box.conf[0])

        if conf < config["confidence_threshold"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # keep coordinates inside image
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)

        # local disparity region
        roi = disparity[cy-5:cy+5, cx-5:cx+5]

        if roi.size == 0:
            continue

        d = np.median(roi)

        if d <= 0:
            continue

        # distance estimation
        distance = (FOCAL_LENGTH * BASELINE) / d
        distance = round(float(distance), 2)

        # visualization color
        color = (0, 0, 255) if distance < 8 else (0, 255, 0)

        # bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # center point
        cv2.circle(
            frame,
            (cx, cy),
            4,
            (255, 0, 0),
            -1
        )

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

    # display frame
    cv2.imshow("Stereo Depth", frame)

    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()