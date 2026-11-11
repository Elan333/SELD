import math

from .nms import calculate_iou


class TrajectoryConfirmationFilter:
    def __init__(self, max_distance, buffer_len, min_hits, grace_frames):
        self.max_distance = max_distance
        self.buffer_len = buffer_len
        self.min_hits = min_hits
        self.grace_frames = grace_frames
        self.history = []
        self.frame_index = 0

    @staticmethod
    def _centroid(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def filter(self, detections):
        self.frame_index += 1
        if self.frame_index <= self.grace_frames or not self.history:
            self.history.append(detections)
            return detections
        confirmed = []
        for detection in detections:
            center = self._centroid(detection["box"])
            hits = 0
            for past in self.history:
                if any(math.dist(center, self._centroid(other["box"])) <= self.max_distance
                       for other in past):
                    hits += 1
            if hits >= self.min_hits:
                confirmed.append(detection)
        self.history.append(detections)
        if len(self.history) > self.buffer_len:
            self.history.pop(0)
        return confirmed


def max_temporal_iou(detections, history):
    values = []
    for detection in detections:
        value = 0.0
        for past in history:
            for other in past:
                value = max(value, calculate_iou(detection["box"], other["box"]))
        values.append(value)
    return values
