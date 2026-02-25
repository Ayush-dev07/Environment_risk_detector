import cv2
from Interference.puddle_detect import ObjectDetector

detector = ObjectDetector()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    detections = detector.detect(frame)

    for d in detections:

        x1,y1,x2,y2 = map(int,d["bbox"])

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            frame,
            f"Puddle {d['confidence']:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("Risk Scanner",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()