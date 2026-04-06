# Environment Risk Detector

A real-time multi-model object detection system that identifies and visualizes environmental hazards including fire, garbage, and puddles using advanced deep learning models.

## Project Overview

The Environment Risk Detector is a computer vision application that leverages three independently trained YOLO (You Only Look Once) models to detect multiple environmental risks simultaneously. The system processes video input from a camera feed and displays real-time detections with confidence scores and visual annotations.

### Key Features

- **Multi-Model Detection**: Simultaneously runs three independent YOLO models for comprehensive risk detection
- **Real-Time Processing**: Processes video frames in real-time with minimal latency
- **Color-Coded Visualization**: Each risk type is distinguished by a unique color for easy identification
- **High Confidence Scoring**: Displays confidence percentages for each detection
- **Flexible Camera Input**: Supports various camera sources and video feeds
- **Independent Risk Detection**: Each model operates independently, allowing detection of multiple risks in the same frame

## Project Structure

```
Environment_risk_detector/
├── main.py                          # Main application file
├── Readme.md                        # Project documentation
├── Risk_detector/                   # Risk detection module
│   ├── fire_detector.py            # Fire detection model
│   ├── garbage_detector.py         # Garbage detection model
│   ├── puddle_detect.py            # Puddle detection model
│   └── __pycache__/
├── Interference/                    # Inference-related utilities
│   ├── detector.py
│   ├── predictor.py
│   ├── tracker.py
│   └── __pycache__/
└── models/                          # Pre-trained model weights
    ├── best.pt                      # Puddle detection model
    ├── fire_best.pt                # Fire detection model
    ├── garbage_best.pt             # Garbage detection model
    └── best/                        # Additional model data
```

## How It Works

### 1. **Initialization Phase**

The application starts by initializing three separate detector objects:

```python
fire_detector = FireDetector()
garbage_detector = GarbageDetector()
puddle_detector = PuddleDetector()
```

Each detector loads its respective pre-trained YOLO model:
- `fire_best.pt` - Trained specifically for fire detection
- `garbage_best.pt` - Trained specifically for garbage/litter detection
- `best.pt` - Trained specifically for puddle/water hazard detection

### 2. **Video Capture**

The system captures video frames from a camera device (camera index 2):
I used camera 2 because using webcam.

### 3. **Real-Time Processing Loop**

For each frame captured from the video feed:

1. **Parallel Detection**: All three models process the frame simultaneously
   - Fire detection
   - Garbage detection
   - Puddle detection

2. **Detection Aggregation**: All detections from the three models are combined into a single list

3. **Result Extraction**: Each detection includes:
   - **Bounding Box**: Coordinates (x1, y1, x2, y2) in pixel coordinates
   - **Confidence Score**: Probability score (0.0 to 1.0) of the detection
   - **Class Label**: Type of risk detected (fire, garbage, puddle)

4. **Visualization**: For each detection:
   - A colored rectangle is drawn around the detected object
   - A label with confidence percentage is displayed above the box
   - Different colors represent different risk types

5. **Display**: The annotated frame is displayed in a window titled "Risk Scanner - All Models"

### 4. **Color Coding**

Each risk type is assigned a unique color in BGR (OpenCV) format:

| Risk Type | Color | RGB Value     |
|-----------|-------|---------------|
| Fire      | Red   | (0, 0, 255)   |
| Garbage   | Orange| (0, 165, 255) |
| Puddle    | Blue  | (255, 0, 0)   |

### 5. **Detection Parameters**

Each detector uses the following parameters for YOLO inference:

- **Confidence Threshold (conf)**: 0.55 - Only detections with confidence ≥ 55% are reported
- **Image Size (imgsz)**: 480 - Input frames are resized to 480x480 pixels for consistency
- **Verbose**: False - Suppresses detailed inference logs

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- OpenCV (cv2)
- YOLOv8 ultralytics library
- A camera device connected to the system

### Required Dependencies

```bash
pip install opencv-python
pip install ultralytics
```

### Model Setup

Ensure the pre-trained model files are present in the `models/` directory:
- `models/fire_best.pt`
- `models/garbage_best.pt`
- `models/best.pt`

## Usage

### Running the Application

```bash
python main.py
```

### Camera Configuration

The application uses camera index 2 by default. If you need to use a different camera:

Edit line in `main.py`:
```python
cap = cv2.VideoCapture(2)
```

Change `2` to your camera index (typically 0 or 1 for default cameras).

### Controls

- **ESC Key**: Press ESC to stop the application and close the video feed
- **Window Close**: Close the "Risk Scanner - All Models" window to exit

## Technical Details

### Object Detection Models

The project uses YOLOv8 (You Only Look Once v8), a state-of-the-art real-time object detection framework:

- **Architecture**: YOLOv8 neural network
- **Input Format**: BGR color images
- **Output**: Bounding boxes with class predictions and confidence scores
- **Training Data**: Each model trained on domain-specific datasets

### Frame Processing Pipeline

1. **Input**: Raw video frame from camera
2. **Resize**: Frame resized to 640x480 for model compatibility
3. **Inference**: YOLO model predicts bounding boxes and classes
4. **Post-Processing**: Results converted to detection dictionaries
5. **Output**: List of detected objects with metadata

## Performance Characteristics

- **Real-Time Processing**: Optimized for live video stream processing
- **Multi-Model Efficiency**: Three models run in sequence on each frame
- **Confidence Threshold**: 0.55 confidence filter reduces false positives
- **Resolution**: 480p processing provides good balance between accuracy and speed

## Error Handling

The application includes basic error handling:

- **Frame Read Failure**: Breaks the loop if video frame cannot be read
- **No Detections**: Gracefully handles frames with no detected risks
- **Keyboard Interrupt**: ESC key cleanly exits and releases camera resource

## Potential Enhancements

1. **Multi-Threading**: Process frames in parallel for improved performance
2. **Tracking**: Add object tracking across frames for persistent identification
3. **Alerts**: Implement audio/visual alerts for high-risk detections
4. **Recording**: Save annotated video output to file
5. **Statistics**: Track and log detection statistics over time
6. **Configuration File**: Allow user-configurable parameters
7. **GPU Support**: Utilize GPU acceleration for faster inference
8. **Multiple Camera Support**: Process feeds from multiple cameras simultaneously

## Troubleshooting

### Camera Not Found
- Check camera connection and index
- List available devices: `v4l2-ctl --list-devices` (on Linux)
- Try different camera indices (0, 1, etc.)

### Low Detection Accuracy
- Adjust confidence threshold in detector files
- Ensure adequate lighting for fire and garbage detection
- Check model file integrity

### Slow Performance
- Reduce input frame resolution in `cv2.resize()`
- Use a GPU-enabled system
- Close other resource-intensive applications

### Model Not Loading
- Verify model files exist in `models/` directory
- Check file permissions
- Ensure ultralytics library is properly installed

## Contributing

To improve the detectors:

1. Collect additional training data for specific risk types
2. Retrain models with new datasets
3. Update model files in the `models/` directory
4. Test with various environmental conditions

## License

This project uses pre-trained YOLO models. Please refer to the Ultralytics YOLOv8 license for usage terms.

## Author

Developed as an environmental hazard detection system for real-time risk assessment and mitigation.

---

For more information about YOLOv8 and the Ultralytics framework, visit: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
