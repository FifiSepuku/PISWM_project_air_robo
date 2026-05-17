import cv2
import numpy as np
from distance.base import DistanceEstimator


class StereoDistanceEstimator(DistanceEstimator):
    # estymacja odległości na podstawie stereo vision (disparity map)

    def __init__(self, focal, baseline, stereo_matcher):
        # parametry kamery i baseline stereo
        self.focal = focal
        self.baseline = baseline

        # obiekt OpenCV Stereo matcher (np. StereoSGBM)
        self.stereo = stereo_matcher

    def estimate(self, frame, detection):
        # konwersja obrazów stereo do skali szarości
        gray_l = cv2.cvtColor(frame.left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame.right, cv2.COLOR_BGR2GRAY)

        # wyznaczenie mapy dysparycji
        disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # środek bounding boxa detekcji
        x1, y1, x2, y2 = detection.bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # lokalny ROI w mapie dysparycji wokół obiektu
        roi = disparity[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5]

        # brak danych w ROI
        if roi.size == 0:
            return None

        # medianowa dysparycja (odporna na szum)
        d = np.median(roi)

        # brak sensownej dysparycji (np. brak korelacji stereo)
        if d <= 0:
            return None

        # wzór stereo: Z = (f * B) / disparity
        distance = (self.focal * self.baseline) / d

        return round(distance, 2)