
import cv2
import os
import json
import numpy as np
from ultralytics import YOLO



# CONFIG
with open("../config/config.json", "r") as f:
    config = json.load(f)

FOCAL_LENGTH = config["focal_length"]
BASELINE = config["baseline"]

model = YOLO(config["model_path"])

left_path = config["left_path"]
right_path = config["right_path"]

# STEREO - ba
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5
)

# PAMIĘĆ TRACKOW
tracks = {}
next_id = 0
MAX_MISSING = 0  # RESET CO KLATKĘ

# KOLOR (BGR MEDIANA)
def get_shirt_color(frame, x1, y1, x2, y2):

    sy1 = y1 + int((y2 - y1) * 0.2)
    sy2 = y1 + int((y2 - y1) * 0.5)

    sx1 = x1 + int((x2 - x1) * 0.2)
    sx2 = x2 - int((x2 - x1) * 0.2)

    roi = frame[sy1:sy2, sx1:sx2]

    if roi.size == 0:
        return (0, 0, 0)

    # mediana
    b = np.median(roi[:, :, 0])
    g = np.median(roi[:, :, 1])
    r = np.median(roi[:, :, 2])

    return (int(b), int(g), int(r))

# MAIN LOOP
for name in os.listdir(left_path):

    if not name.endswith(".png"):
        continue

    left = cv2.imread(os.path.join(left_path, name))
    right = cv2.imread(os.path.join(right_path, name))

    if left is None or right is None:
        continue

    frame = left.copy()
    h, w = frame.shape[:2]

    
    # UI PANEL
    panel_w = 320
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    
    # GLEBIA STEREO
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # zmiana typu danych z int16 na float32 w celu umożliwienia operacji zmiennoprzecinkowych
    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0


    
    # YOLO - w klasie 0 są piesi
    results = model(left, classes=[0], conf=config["confidence_threshold"])
    boxes = results[0].boxes
    detections = []
    active_ids = set()

    # DETEKCJE
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

    # SLEDZENIE
        for x1, y1, x2, y2, cx, cy, Z, color in detections:

            best_id = None
            best_score = 1e9

            for tid, t in tracks.items():

                px, py = t["centroid"]
                pz = t["Z"]
                pc = np.array(t["color"])

                
                # NORMALIZACJA SKŁADOWYCH
                # 1) POZYCJA (0–1) current - previous, odległość liniowa
                pos_dist = np.linalg.norm([cx - px, cy - py])
                # jeżeli różnica w centrach pozycji jest większa niż 150px to
                # dopasowanie jest uznawane za bardzo złe - duża kara
                pos_score = min(pos_dist / 150.0, 1.0)

                # 2) ODLEGŁOŚĆ Z (0–1)
                dz = abs(Z - pz)
                # jeżeli różnica w dystansie od kamery jest większa
                # niż 3m to dopasowanie jest uznawane za bardzo złe
                z_score = min(dz / 3.0, 1.0)

                # 3) KOLOR BGR (0–1) na podstawie dystansu euklidesowego
                dc = np.linalg.norm(np.array(color) - pc)
                color_score = min(dc / 441.0, 1.0)

                
                # WAGI dopasowane na oko
                score = (
                        0.05 * pos_score +
                        0.05 * z_score +
                        0.90 * color_score
                )

                if pos_dist < 150 and dz < 2.0 and score < best_score:
                    best_score = score
                    best_id = tid
        
        # NOWE SLEDZENIE
        if best_id is None:

            tid = next_id
            next_id += 1

            tracks[tid] = {
                "centroid": (cx, cy),
                "bbox": (x1, y1, x2, y2),
                "Z": Z,
                "color": color,
                "missing": 0
            }

            best_id = tid

        
        # AKTUALIZACJA
        else:

            tracks[best_id]["centroid"] = (cx, cy)
            tracks[best_id]["bbox"] = (x1, y1, x2, y2)
            tracks[best_id]["Z"] = Z
            tracks[best_id]["color"] = color
            tracks[best_id]["missing"] = 0

        active_ids.add(best_id)

        
        # BOX PIESZEGO
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)


    
    # USUWANIE TRACKOW
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


    
    # PANEL + MAPA POZYCJI
    
    panel_positions = {}
    y_offset = 40

    for tid, t in tracks.items():

        x_panel = w + 10
        y_panel = y_offset

        panel_positions[tid] = (x_panel, y_panel)

        cx, cy = t["centroid"]
        b, g, r = t["color"]

        cv2.rectangle(
            canvas,
            (w, y_panel - 25),
            (w + panel_w, y_panel + 60),
            (30, 30, 30),
            -1
        )

        cv2.putText(canvas, f"ID: {tid}", (x_panel, y_panel),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(canvas, f"PX: {cx},{cy}", (x_panel, y_panel + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(canvas, f"Z: {t['Z']:.2f}m", (x_panel, y_panel + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(canvas, f"BGR: {b},{g},{r}", (x_panel, y_panel + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset += 90


    
    # LINIE WSKAZNIKOW
    for tid, t in tracks.items():

        if tid not in panel_positions:
            continue

        px, py = t["centroid"]
        panel_x, panel_y = panel_positions[tid]

        cv2.line(
            canvas,
            (px, py),
            (panel_x, panel_y),
            t["color"],
            2
        )


    
    # SHOW
    
    cv2.imshow("RGB Stable Tracking UI", canvas)

    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()