import cv2
import numpy as np

def detect_anomalies(frame1, frame2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference between the two images
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold the difference to get binary image
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours of the anomalies
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original frame
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame1

# Initialize video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Read the first frame to initialize the previous frame
ret, prev_frame = cap.read()

while True:
    # Read the current frame
    ret, curr_frame = cap.read()
    if not ret:
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
