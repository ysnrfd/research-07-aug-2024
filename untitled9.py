#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 21:01:48 2024

@author: ysnrfd
"""

import cv2
import numpy as np

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Create background subtractor with KNN
    backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=.512, detectShadows=True)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    try:
        while True:
            # Read frame from the camera
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read frame.")
                break

            # Apply background subtraction
            fgMask = backSub.apply(frame)

            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(fgMask, (5, 5), 0)

            # Find contours
            contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes around detected objects
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter out small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the results
            cv2.imshow('Frame', frame)
            cv2.imshow('Foreground Mask', fgMask)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
