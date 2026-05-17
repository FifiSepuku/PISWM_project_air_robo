from dataclasses import dataclass

# struktura danych opisująca pojedynczą detekcję człowieka
@dataclass
class Detection:
    # współrzędne bounding boxa: (x1, y1, x2, y2)
    bbox: tuple
    # pewność detekcji modelu YOLO (0–1)
    confidence: float
    # wyliczona odległość od kamery (w metrach), opcjonalna
    distance: float | None = None
    # poziom ryzyka (np. LOW / MEDIUM / HIGH), opcjonalny
    risk: str | None = None