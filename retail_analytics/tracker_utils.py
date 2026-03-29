import numpy as np

class PathTracker:
    def __init__(self):
        self.paths = {}

    def update(self, track_id, centroid):
        if track_id not in self.paths:
            self.paths[track_id] = []
        self.paths[track_id].append(centroid)

    def get_paths(self):
        return self.paths
    