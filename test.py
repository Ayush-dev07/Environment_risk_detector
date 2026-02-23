import cv2
import csv
import os

from Interference.detector import YOLODetector
from Interference.tracker import CentroidTracker


# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = "/home/ayush02/Downloads/6145681-uhd_2160_3840_24fps.mp4"
OUTPUT_CSV = "data/trajectories.csv"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# -----------------------------
# INIT
# -----------------------------
detector = YOLODetector(conf=0.4)
tracker = CentroidTracker()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video")
    exit()

os.makedirs("data", exist_ok=True)

csv_file = open(OUTPUT_CSV, "w", newline="")
writer = csv.writer(csv_file)

# CSV HEADER
writer.writerow(["video","object_id","frame","x","y"])


frame_id = 0
video_name = os.path.basename(VIDEO_PATH)

print("Processing video... Press Q to stop.")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Resize for speed
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # ---------------- DETECTION
    detections = detector.detect(frame)

    # ---------------- TRACKING
    objects = tracker.update(detections)

    # ---------------- LOG DATA
    for obj_id, (cx, cy) in objects.items():

        # Normalize coordinates
        x_norm = cx / FRAME_WIDTH
        y_norm = cy / FRAME_HEIGHT

        writer.writerow([
            video_name,
            obj_id,
            frame_id,
            x_norm,
            y_norm
        ])

        # Draw on screen
        cv2.circle(frame,(cx,cy),5,(0,255,0),-1)
        cv2.putText(frame,f"ID {obj_id}",
                    (cx-10,cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,255,0),2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
csv_file.close()
cv2.destroyAllWindows()

print("Done. Trajectories saved.")
