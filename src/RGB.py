import cv2
import os
import json
import numpy as np
from ultralytics import YOLO


# =========================
# CONFIG
# =========================
with open("../config/config.json", "r") as f:
    config = json.load(f)

FOCAL_LENGTH = config["focal_length"]
BASELINE = config["baseline"]

model = YOLO(config["model_path"])

left_path = config["left_path"]
right_path = config["right_path"]


# =========================
# STEREO
# =========================
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5
)


# =========================
# TRACKS
# =========================
tracks = {}
next_id = 0
MAX_MISSING = 2


# =========================
# COLOR (BGR MEDIAN)
# =========================
def get_shirt_color(frame, x1, y1, x2, y2):

    sy1 = y1 + int((y2 - y1) * 0.2)
    sy2 = y1 + int((y2 - y1) * 0.5)

    sx1 = x1 + int((x2 - x1) * 0.2)
    sx2 = x2 - int((x2 - x1) * 0.2)

    roi = frame[sy1:sy2, sx1:sx2]

    if roi.size == 0:
        return (0, 0, 0)

    b = np.median(roi[:, :, 0])
    g = np.median(roi[:, :, 1])
    r = np.median(roi[:, :, 2])

    return (int(b), int(g), int(r))


# =========================
# MAIN LOOP
# =========================
for name in os.listdir(left_path):

    if not name.endswith(".png"):
        continue

    left = cv2.imread(os.path.join(left_path, name))
    right = cv2.imread(os.path.join(right_path, name))

    if left is None or right is None:
        continue

    frame = left.copy()
    h, w = frame.shape[:2]

    panel_w = 320
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    # =========================
    # DEPTH
    # =========================
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0


    # =========================
    # DETECTION
    # =========================
    results = model(left, classes=[0], conf=config["confidence_threshold"])
    boxes = results[0].boxes

    detections = []
    active_ids = set()


    # =========================
    # BUILD DETECTIONS
    # =========================
    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        roi = disparity[max(0, cy-5):cy+5, max(0, cx-5):cx+5]

        if roi.size == 0:
            continue

        d = np.median(roi)
        if d <= 0:
            continue

        Z = (FOCAL_LENGTH * BASELINE) / d
        color = get_shirt_color(left, x1, y1, x2, y2)

        detections.append((x1, y1, x2, y2, cx, cy, Z, color))


    # =========================
    # MATCHING (STABLE)
    # =========================
    for x1, y1, x2, y2, cx, cy, Z, color in detections:

        best_id = None
        best_score = 1e9

        for tid, t in tracks.items():

            px, py = t["centroid"]
            pz = t["Z"]
            pc = np.array(t["color"])

            pos_dist = np.linalg.norm([cx - px, cy - py])
            dz = abs(Z - pz)
            dc = np.linalg.norm(np.array(color) - pc)

            score = (
                0.30 * min(pos_dist / 150.0, 1.0) +
                0.30 * min(dz / 3.0, 1.0) +
                0.40 * min(dc / 441.0, 1.0)
            )

            if pos_dist < 150 and dz < 2.0 and score < best_score:
                best_score = score
                best_id = tid


        # =========================
        # NEW TRACK
        # =========================
        if best_id is None:

            tid = next_id
            next_id += 1

            tracks[tid] = {
                "centroid": (cx, cy),
                "Z": Z,
                "color": color,
                "history": [(cx, cy, Z)],
                "missing": 0
            }

            best_id = tid

        # =========================
        # UPDATE TRACK
        # =========================
        else:

            t = tracks[best_id]

            t["centroid"] = (cx, cy)
            t["Z"] = Z
            t["color"] = color

            t["history"].append((cx, cy, Z))
            if len(t["history"]) > 2:
                t["history"].pop(0)

            t["missing"] = 0

        active_ids.add(best_id)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # =========================
    # REMOVE LOST TRACKS
    # =========================
    to_delete = []

    for tid, t in tracks.items():

        if tid in active_ids:
            t["missing"] = 0
        else:
            t["missing"] += 1

        if t["missing"] > MAX_MISSING:
            to_delete.append(tid)

    for tid in to_delete:
        del tracks[tid]


    # =========================
    # PANEL POSITIONS
    # =========================
    panel_positions = {}
    y_offset = 40

    for tid, t in tracks.items():

        panel_positions[tid] = (w + 10, y_offset)

        cx, cy, z = *t["centroid"], t["Z"]

        cv2.rectangle(canvas,
                      (w, y_offset - 25),
                      (w + panel_w, y_offset + 60),
                      (30, 30, 30),
                      -1)

        cv2.putText(canvas, f"ID {tid}", (w + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(canvas, f"Z {z:.2f}m", (w + 10, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 90


    # =========================
    # PREDICTION + VECTOR
    # =========================
    for tid, t in tracks.items():

        if len(t["history"]) < 2:
            continue

        (x1, y1, z1), (x2, y2, z2) = t["history"]

        vx = x2 - x1
        vy = y2 - y1
        vz = z2 - z1

        cx, cy = t["centroid"]

        px = int(cx + vx * 3)
        py = int(cy + vy * 3)

        # current
        cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1)

        # prediction
        cv2.circle(canvas, (px, py), 5, (0, 255, 255), 2)

        # vector
        cv2.line(canvas, (cx, cy), (px, py), (0, 255, 255), 2)

        # panel link
        if tid in panel_positions:
            panel_x, panel_y = panel_positions[tid]
            cv2.line(canvas, (cx, cy), (panel_x, panel_y), t["color"], 1)


    # =========================
    # SHOW
    # =========================
    cv2.imshow("Stable 3D Tracking + Prediction", canvas)

    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()