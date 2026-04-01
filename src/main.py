import cv2
import os
import natsort

path = r"/home/filip/projekt_pism/train/chosen_data/19-10-2018_13-33-29/RGB"
image_files = [f for f in os.listdir(path) if f.endswith(".png")]
image_files = natsort.natsorted(image_files)

for img_name in image_files:

    img_path = os.path.join(path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print("Nie wczytano:", img_path)
        continue

    cv2.imshow("Image View", img)

    if cv2.waitKey(17) == 27:
        break

cv2.destroyAllWindows()