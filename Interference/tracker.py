import numpy as np
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id =+ 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1

                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects
        
        input_centroids = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cX = int((x1+x2)/2.0)
            cY = int((y1+y2)/2.0)
            input_centroids.append((cX, cY))

        input_centroids = np.array(input_centroids)

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(object_centroids)[:, None] - input_centroids,
                axis = 2
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols

            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1

                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister[obj_id]

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects
    