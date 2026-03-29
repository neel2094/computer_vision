import numpy as np
import cv2

class Heatmap:
    def __init__(self, width, height):
        self.heatmap = np.zeros((height, width), dtype=np.float32)

    def update(self, x, y):
        if 0 <= x < self.heatmap.shape[1] and 0 <= y < self.heatmap.shape[0]:
            self.heatmap[int(y), int(x)] += 1

    def get_colored(self):
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype('uint8'), cv2.COLORMAP_JET)
        return colored
