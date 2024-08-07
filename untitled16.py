#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:48:50 2024

@author: ysnrfd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Ghost Detection Script with Machine Learning Integration and Real-Time Performance Enhancements
"""

import cv2
import numpy as np
from threading import Thread
from queue import Queue
from ultralytics import YOLO

class AdvancedGhostDetector:
    def __init__(self, video_source=0, contour_area_threshold=100, model_path='yolov8s.pt'):
        self.video_source = video_source
        self.contour_area_threshold = contour_area_threshold
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.cap = cv2.VideoCapture(self.video_source)
        self.frame_queue = Queue(maxsize=10)
        self.model = YOLO(model_path)

        if not self.cap.isOpened():
            raise IOError("Error: Could not open camera.")

    def capture_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def process_frame(self, frame):
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Find contours of the detected objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the frame
        for contour in contours:
            if cv2.contourArea(contour) > self.contour_area_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Anomaly Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def detect_objects(self, frame):
        # Convert frame to RGB for YOLOv8
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_frame, imgsz=320)

        # Draw bounding boxes and labels on the frame
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, score, class_id = map(int, box)
                label = f"{self.model.names[class_id]}: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def display_frame(self, frame):
        # Display the resulting frame
        cv2.imshow('Advanced Ghost Detector', frame)

    def run(self):
        capture_thread = Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()

        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame = self.process_frame(frame)
                detected_frame = self.detect_objects(processed_frame)
                self.display_frame(detected_frame)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cleanup()

    def cleanup(self):
        # Release the capture and close all windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = AdvancedGhostDetector(video_source=0, contour_area_threshold=1000, model_path='yolov8n.pt')
        detector.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        cv2.destroyAllWindows()
