import cv2
import os
from ultralytics import YOLO

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
    results = model(img)

    # obraz z bounding boxami
    frame = results[0].plot()

    # liczenie pieszych
    boxes = results[0].boxes
    people_count = 0

    for box in boxes:
        cls = int(box.cls[0])

        if cls == 0:  # klasa 0 = person
            people_count += 1

    print(f"{img_name} -> pedestrians: {people_count}")

    cv2.imshow("YOLO detection", frame)

    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()