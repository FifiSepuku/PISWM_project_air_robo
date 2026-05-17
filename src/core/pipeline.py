class Pipeline:
    # główny moduł przetwarzania obrazu
    # łączy detekcję, estymację odległości, ocenę ryzyka i wizualizację

    def __init__(self, detector, estimator, risk, renderer):
        # komponent wykrywający obiekty (np. YOLO)
        self.detector = detector

        # komponent liczący odległość (stereo / depth)
        self.estimator = estimator

        # komponent oceny ryzyka na podstawie dystansu
        self.risk = risk

        # komponent odpowiedzialny za rysowanie wyników na obrazie
        self.renderer = renderer

    def process(self, frame):
        # etap 1: detekcja obiektów na obrazie
        detections = self.detector.detect(frame)

        # etap 2: wzbogacenie detekcji o odległość i ryzyko
        for d in detections:
            d.distance = self.estimator.estimate(frame, d)
            d.risk = self.risk.assess(d.distance)

        # etap 3: wizualizacja wyników
        return self.renderer.draw(frame, detections)