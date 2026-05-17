class DistanceEstimator:
    # klasa bazowa (interfejs) dla estymacji odległości
    # definiuje wspólny kontrakt dla różnych metod (stereo, depth)

    def estimate(self, frame, detection):
        # metoda abstrakcyjna
        # powinna zostać nadpisana w klasach dziedziczących
        # np. StereoDistanceEstimator, DepthDistanceEstimator
        raise NotImplementedError