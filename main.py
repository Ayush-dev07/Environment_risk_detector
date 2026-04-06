import cv2
from Risk_detector.fire_detector import ObjectDetector as FireDetector
from Risk_detector.garbage_detector import ObjectDetector as GarbageDetector
from Risk_detector.puddle_detect import ObjectDetector as PuddleDetector

fire_detector = FireDetector()
garbage_detector = GarbageDetector()
puddle_detector = PuddleDetector()

colors = {
    "fire": (0, 0, 255),      # Red
    "garbage": (0, 165, 255),  # Orange
    "puddle": (255, 0, 0)      # Blue
}

cap = cv2.VideoCapture(2)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    fire_detections = fire_detector.detect(frame)
    garbage_detections = garbage_detector.detect(frame)
    puddle_detections = puddle_detector.detect(frame)

    all_detections = fire_detections + garbage_detections + puddle_detections

    for d in all_detections:

        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d["label"]
        confidence = d["confidence"]
        color = colors[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label.upper()} {confidence:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Risk Scanner - All Models", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()