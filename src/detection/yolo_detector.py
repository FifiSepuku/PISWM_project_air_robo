from core.detection import Detection


class YoloDetector:
    # klasa odpowiedzialna za detekcję ludzi przy użyciu modelu YOLO

    def __init__(self, model, conf):
        # model YOLO (np. ultralytics.YOLO)
        self.model = model

        # próg pewności detekcji
        self.conf = conf

    def detect(self, frame):
        # wykonanie detekcji na obrazie RGB
        results = self.model(frame.rgb, classes=[0], conf=self.conf)
        boxes = results[0].boxes

        detections = []

        # przetwarzanie wyników YOLO na obiekty Detection
        for box in boxes:
            conf = float(box.conf[0])

            # filtracja słabych detekcji
            if conf < self.conf:
                continue

            # bounding box (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf
                )
            )

        return detections