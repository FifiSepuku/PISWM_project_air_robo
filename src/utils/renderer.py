import cv2


class Renderer:
    # klasa odpowiedzialna za wizualizację wyników pipeline’u na obrazie

    def draw(self, frame, detections):
        # kopia obrazu wejściowego (żeby nie modyfikować oryginału)
        img = frame.rgb.copy()

        # iteracja po wszystkich detekcjach
        for d in detections:
            x1, y1, x2, y2 = d.bbox

            # wybór koloru na podstawie dystansu (heurystyka zagrożenia)
            # czerwony = blisko, zielony = bezpiecznie
            color = (0, 0, 255) if d.distance and d.distance < 8 else (0, 255, 0)

            # rysowanie bounding boxa
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # rysowanie tekstu tylko jeśli mamy estymację odległości
            if d.distance:
                cv2.putText(
                    img,
                    f"{d.distance:.2f} m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # zwrócenie obrazu z naniesionymi wynikami
        return img