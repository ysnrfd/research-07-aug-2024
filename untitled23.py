#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:11:43 2024

@author: ysnrfd
"""

import cv2
import numpy as np

def detect_anomalies(frame1, frame2, min_contour_area=1, threshold_value=8, blur_ksize=(5, 5)):
    """
    Detects anomalies between two frames with high sensitivity.

    Parameters:
    - frame1: The previous frame.
    - frame2: The current frame.
    - min_contour_area: Minimum area for a contour to be considered an anomaly.
    - threshold_value: Threshold value for binary conversion.
    - blur_ksize: Kernel size for Gaussian blur.

    Returns:
    - The frame with anomalies highlighted.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the two images
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply GaussianBlur to reduce noise and improve thresholding
    blurred = cv2.GaussianBlur(diff, blur_ksize, 0)
    
    # Threshold the difference to get a binary image
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours of the anomalies
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original frame
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 1)
    
    return frame1

def main():
    # Initialize video capture (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Read the first frame to initialize the previous frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        cap.release()
        return

    while True:
        # Read the current frame
        ret, curr_frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Detect anomalies between previous and current frame
        result_frame = detect_anomalies(prev_frame, curr_frame)
        
        # Display the result
        cv2.imshow('Anomalies Detected', result_frame)
        
        # Update previous frame
        prev_frame = curr_frame
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
