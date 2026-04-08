import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

people_counts = []

# wczytanie modelu YOLO
model = YOLO("yolov8n.pt")

path = r"G:\Python\Synthia\Stereo_Left\Omni_F"

for img_name in os.listdir(path):

    if not img_name.lower().endswith(".png"):
        continue

    img_path = os.path.join(path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # detekcja
    results = model(img, classes=[0])

    # obraz z bounding boxami
    frame = results[0].plot()

    # liczenie pieszych
    boxes = results[0].boxes
    people_count = 0

    h, w = img.shape[:2]
    screen_center = w // 2

    for box in boxes:
        cls = int(box.cls[0])

        if cls == 0:  # person
            people_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            height = y2 - y1

            # rysowanie środka
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # WARNING: blisko środka
            if abs(center_x - screen_center) < 100:
                cv2.putText(frame,
                            "WARNING: CENTER",
                            (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2)

            # duży obiekt = blisko
            if height > 0.3 * h:
                cv2.putText(frame,
                            "COLLISION RISK",
                            (40, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 0, 255),
                            3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # ostrzeżenie: wykryto pieszego
    if people_count > 0:
        cv2.putText(frame,
                    "PEDESTRIAN DETECTED",
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

    print(f"{img_name} -> pedestrians: {people_count}")
    people_counts.append(people_count)

    cv2.imshow("YOLO detection", frame)

    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()

# wykres
plt.plot(people_counts)
plt.title("Number of pedestrians per frame")
plt.xlabel("Frame")
plt.ylabel("People count")
plt.grid()
plt.show()