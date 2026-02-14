from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf)
        detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, score , cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)

                detections.append({
                    "bbox": (x1,y1,x2,y2),
                    "score": float(score),
                    "class_id": int(cls)
                })
        return detections
    
if __name__=="__main__":
    cap = cv2.VideoCapture(2)

    detector = YOLODetector()

    while True:
        ret, frame =cap.read()
        frame = cv2.resize(frame, (416, 416))
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            cv2.rectangle(frame, (x1,y1), (x2, y2),
                          (0, 255, 0), 2)
            
        cv2.imshow("YOLO Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
