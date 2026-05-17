import json
import cv2
import os
from ultralytics import YOLO

# import własnych komponentów systemu
from core.frame import ImageFrame
from core.pipeline import Pipeline
from detection.yolo_detector import YoloDetector
from utils.risk import RiskAssessor
from utils.renderer import Renderer

from distance.stereo import StereoDistanceEstimator
from distance.depth import DepthDistanceEstimator


# wczytanie konfiguracji z pliku JSON
with open("../config/config.json") as f:
    config = json.load(f)

# załadowanie modelu YOLO do detekcji ludzi
model = YOLO(config["model_path"])

# inicjalizacja detektora (YOLO + próg pewności)
detector = YoloDetector(model, config["confidence_threshold"])

# wybór metody estymacji odległości na podstawie konfiguracji
if config["mode"] == "stereo":
    # inicjalizacja stereo vision (mapa dysparycji)
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5)

    # estymator odległości na podstawie stereo
    estimator = StereoDistanceEstimator(
        config["focal_length"],
        config["baseline"],
        stereo
    )
else:
    # estymator odległości na podstawie mapy depth
    estimator = DepthDistanceEstimator()

# złożenie całego pipeline’u przetwarzania obrazu
pipeline = Pipeline(
    detector=detector,
    estimator=estimator,
    risk=RiskAssessor(config["collision_threshold"]),
    renderer=Renderer()
)

# główna pętla przetwarzania danych (kolejne klatki obrazu)
for name in os.listdir(config["left_path"]):

    # wczytanie obrazu lewego (stereo / RGB)
    left = cv2.imread(os.path.join(config["left_path"], name))

    # wczytanie obrazu prawego (stereo)
    right = cv2.imread(os.path.join(config["right_path"], name))

    # wczytanie mapy depth (dla trybu depth-based)
    depth = cv2.imread(os.path.join(config["depth_path"], name))

    # stworzenie obiektu reprezentującego jedną klatkę danych
    frame = ImageFrame(rgb=left, depth=depth, left=left, right=right)

    # przetworzenie klatki przez cały pipeline (detekcja - dystans - ryzyko - wizualizacja)
    output = pipeline.process(frame)

    # wyświetlenie wyniku
    cv2.imshow("result", output)

    # wyjście po wciśnięciu ESC
    if cv2.waitKey(50) == 27:
        break