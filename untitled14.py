#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:35:09 2024

@author: ysnrfd
"""

import cv2
from ultralytics import YOLO

# Load the YOLOv8n model (nano) for ultra-fast inference
model = YOLO('yolov10n.pt')  # Replace with the path to your YOLOv8n model

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the camera resolution (lower resolution for speed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Perform object detection with YOLOv8n
    results = model(frame, imgsz=512, stream=True)  # Adjust img size if necessary

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, score, class_id = map(int, box)
            label = f"{model.names[class_id]}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Thin box for speed
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Thin text for speed

    # Display the resulting frame
    cv2.imshow('YOLOv8n Real-Time Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
