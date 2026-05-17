class RiskAssessor:
    # klasa oceniająca poziom ryzyka na podstawie odległości

    def __init__(self, threshold):
        # próg odległości (poniżej = ryzyko wysokie)
        self.threshold = threshold

    def assess(self, distance):
        # brak danych o odległości → nie można ocenić ryzyka
        if distance is None:
            return "UNKNOWN"

        # obiekt zbyt blisko → wysokie ryzyko
        if distance < self.threshold:
            return "HIGH"

        # wszystko OK
        return "LOW"