import cv2
import os
import natsort
from ultralytics import YOLO

# wczytanie modelu YOLO
model = YOLO("yolov8n.pt")

path = r"/home/filip/projekt_pism/train/chosen_data/19-10-2018_13-33-29/RGB"
image_files = [f for f in os.listdir(path) if f.endswith(".png")]
image_files = natsort.natsorted(image_files)

for img_name in image_files:


    img_path = os.path.join(path, img_name)

    img = cv2.imread(img_path)

    if img is None:
        print("Nie wczytano:", img_path)
        continue

    # detekcja tylko ludzi i z pewnoscia 60%
    results = model(img, classes = [0],conf = 0.6, verbose=False)

    # obraz z bounding boxami
    frame = results[0].plot()

    # liczenie pieszych
    people_count = len(results[0].boxes)

    print(f"{img_name} -> pedestrians: {people_count}")

    cv2.imshow("YOLO detection", frame)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()