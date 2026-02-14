import cv2
import numpy as np

class KalmanTrajectoryPredictor:
    def __init__(self):
        self.filters = {}

    def _create_kf(self):
        kf = cv2.KalmanFilter(4,2)

        kf.transitionMatrix = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]
        ], np.float32)

        kf.measurementMatrix = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ], np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32)*0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32)*0.5

        return kf
    
    def update(self, objects):
        predictions = {}

        for obj_id, (x,y) in objects.items():
            if obj_id not in self.filters:
                kf = self._create_kf()
                kf.statePre = np.array([[x], [y], [0], [0]],
                                       dtype=np.float32)
                self.filters[obj_id] = kf

            kf =self.filters[obj_id]

            pred = kf.predict()

            measurements = np.array([[np.float32(x)],
                                     [np.float32(y)]])
            kf.correct(measurements)

            predictions[obj_id] = (int(pred[0]),
                                   int(pred[1]))
            
        return predictions
    
    def predict_future(self, obj_id, steps=5):
        if obj_id not in self.filters:
            return None
        
        kf = self.filters[obj_id]

        future_pts = []
        temp_state = kf.statePost.copy()

        for _ in range(steps):
            temp_state = kf.transitionMatrix @ temp_state
            future_pts.append((
                int(temp_state[0]),
                int(temp_state[1])
            ))

        return future_pts