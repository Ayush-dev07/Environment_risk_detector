from ultralytics import YOLO
import cv2

class ObjectDetector:

    def __init__(self):

        # Load your trained model
        self.model = YOLO("models/garbage_best.pt")

    def detect(self, frame):

        small_frame = cv2.resize(frame, (640, 480))

        results = self.model.predict(
            small_frame,
            conf=0.55,
            imgsz=480,
            verbose=False
        )

        detections = []

        for r in results:

            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    "bbox":[x1,y1,x2,y2],
                    "confidence":conf,
                    "class":cls,
                    "label":"garbage"
                })

        return detections