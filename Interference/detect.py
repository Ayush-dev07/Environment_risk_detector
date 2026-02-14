import cv2
from ultralytics import YOLO

model = YOLO("models/trained/best.pt")

def risk_label(conf):
    if conf > 0.75:
        return "HIGH RISK"
    elif conf > 0.5:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not accessible")
    exit()

print("Real-time risk detection started")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=320, verbose=False)

    annotated = results[0].plot()

    for box in results[0].boxes:
        conf = float(box.conf[0])
        label = risk_label(conf)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Environment Risk Scanner", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
