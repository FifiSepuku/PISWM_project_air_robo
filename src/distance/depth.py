import numpy as np
from distance.base import DistanceEstimator


class DepthDistanceEstimator(DistanceEstimator):
    # estymator odległości bazujący na mapie depth

    def __init__(self, roi_size=30):
        # rozmiar okna (ROI) wokół środka detekcji
        self.roi_size = roi_size

    def estimate(self, frame, detection):
        # pobranie bboxa wykrytego obiektu
        x1, y1, x2, y2 = detection.bbox

        # wyznaczenie środka obiektu
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # rozmiar obrazu depth
        h, w = frame.depth.shape[:2]

        # wyznaczenie regionu zainteresowania wokół środka
        x_min = max(cx - self.roi_size, 0)
        x_max = min(cx + self.roi_size, w)
        y_min = max(cy - self.roi_size, 0)
        y_max = min(cy + self.roi_size, h)

        roi = frame.depth[y_min:y_max, x_min:x_max]

        # zabezpieczenie przed pustym ROI
        if roi.size == 0:
            return None

        # konwersja do float (stabilność obliczeń)
        depth_values = roi.astype(np.float32)

        # usunięcie niepoprawnych wartości (0 = brak pomiaru)
        depth_values = depth_values[depth_values > 0]

        # brak sensownych danych
        if len(depth_values) == 0:
            return None

        # odporna statystycznie estymacja odległości
        return float(np.median(depth_values))