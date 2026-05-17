class ImageFrame:
    # klasa przechowująca dane jednego kroku przetwarzania obrazu

    def __init__(self, rgb, depth=None, left=None, right=None):
             # obraz RGB
            self.rgb = rgb
             # mapa głębi (dla wariantu depth-based)
            self.depth = depth
             # obraz lewy (dla stereo vision)
            self.left = left
             # obraz prawy (dla stereo vision)
            self.right = right
            # lista wykrytych obiektów (Detection)
            self.detections = []