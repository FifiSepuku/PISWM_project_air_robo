import cv2
import os

path = r"G:\Python\Synthia\Stereo_Left\Omni_F"

for img_name in os.listdir(path):

    img_path = os.path.join(path, img_name)

    img = cv2.imread(img_path)

    if img is None:
        print("Nie wczytano:", img_path)
        continue

    cv2.imshow("Preview", img)

    if cv2.waitKey(100) == 27:
        break

cv2.destroyAllWindows()