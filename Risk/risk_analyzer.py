import numpy as np

class RiskAnalyzer:
    def __init__(self, frame_width, frame_height, danger_radius=120):
        self.fw = frame_width
        self.fh = frame_height

        self.user_pos = np.array(
            [frame_width//2, frame_height//2]
        )
        self.danger_radius = danger_radius
        self.prev_positions = {}

    def analyze(self, object_positions):
        risks = {}

        for obj_id, pos in object_positions.items():
            curr = np.array(pos)
            dist = np.linalg.norm(curr-self.user_pos)
            if obj_id not in self.prev_positions:
                self.prev_positions[obj_id]=curr
                continue
            prev = self.prev_positions[obj_id]
            velocity = curr - prev
            speed = np.linalg.norm(velocity)

            if speed == 0:
                continue

            to_user = self.user_pos - curr
            dot = np.dot(velocity, to_user)

            moving_towards = dot > 0

            ttc = dist/(speed + 1e-5)
            risk_level = "LOW"

            if moving_towards:
                if ttc < 1.0:
                    risk_level = "HIGH"
                elif ttc < 2.0:
                    risk_level = "MEDIUM"

            if dist < self.danger_radius:
                risk_level = "HIGH"

            risks[obj_id] = {
                "distance" : float(dist),
                "speed" : float(speed),
                "ttc" : float(ttc),
                "risk" : risk_level
            }

            self.prev_positions[obj_id] = curr
        return risks
    