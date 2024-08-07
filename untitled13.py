import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8x model and move it to GPU if available
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for even faster processing if accuracy is acceptable

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the camera resolution (lower resolution for faster processing)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced resolution for speed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert to grayscale for night vision effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a color map to simulate night vision
    night_vision = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

    # Convert the frame to the format required by YOLOv8
    night_vision_rgb = cv2.cvtColor(night_vision, cv2.COLOR_BGR2RGB)

    # Perform object detection with YOLOv8x
    results = model(night_vision_rgb, stream=True, imgsz=320)  # Further reduced image size for speed

    # Draw bounding boxes and labels on the night vision image
    for result in results:
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, score, class_id = map(int, box)
            label = f"{model.names[class_id]}: {score:.2f}"
            cv2.rectangle(night_vision, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Thin box for speed
            cv2.putText(night_vision, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Thin text for speed

    # Display the resulting frame
    cv2.imshow('Night Vision YOLOv8x', night_vision)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
