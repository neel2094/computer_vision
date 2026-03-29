class LoiteringDetector:
    def __init__(self, fps, threshold_seconds=10):
        self.time_inside = {}
        self.threshold_frames = fps * threshold_seconds

    def update(self, track_id, inside_roi):
        if track_id not in self.time_inside:
            self.time_inside[track_id] = 0

        if inside_roi:
            self.time_inside[track_id] += 1
        else:
            self.time_inside[track_id] = 0

        return self.time_inside[track_id] > self.threshold_frames
