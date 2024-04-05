import argparse
from importlib import reload
from ultralytics import YOLO
from functions.hough_functions import LineDetector
from functions import other_functions
from collections import defaultdict

import numpy as np
import pandas as pd
import csv
import cv2

parser = argparse.ArgumentParser(description='Detect objects in an image and save results to a CSV file.')
parser.add_argument('--image', '-i', type=str, default='inputs/train_clip.mp4', help='Path to the input image file')
args = parser.parse_args()

################################################

# BUILD / PULL MODEL

################################################

model = YOLO('models/yolov8n.pt')
#results = model.train(data='coco128.yaml', epochs=3)
#results = model.val()

################################################

# LOAD IMAGE AND DEFINE VARIABLES

################################################

# open video file
video_1 = args.image
cap = cv2.VideoCapture(video_1)

# Prep the CSV file
csv_file_name = "output/video_objects.csv"
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame','Class ID', 'Object Name','Confidence', 'X1', 'Y1', 'X2', 'Y2', 'Distance'])

# Define the codec and create VideoWriter object
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output/output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize a list to store detected objects
detected_objects = []
i = 0

################################################

# LOOP THROUGH FRAMES

################################################

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        
        
        ################################################

        # LOAD IMAGE AND APPLY MODELs

        ################################################

        # Load the image
        image = frame

        # Apply the YOLO object detection model
        results = model(image)

        # Apply the Hough Line Transform
        detector = LineDetector()
        lines = detector.detect_lines_frame(image)

        # Calculate Distances
        distance = 0
        #distance, direction = other_functions.putting_it_all_together(lines, 1700, 1000, 9.5)
        
        
        ################################################

        # WRITE RESULTS TO CSV

        ################################################
        
        detected_objects = results[0].boxes
        object_names = results[0].names
        with open(csv_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            for box in detected_objects:
                class_id = box.cls[0].item()
                conf = box.conf[0].item()
                cords = box.xyxy[0].tolist()  # formatted as [x1, y1, x2, y2]
                object_name = object_names.get(class_id, 'Unknown')

                # Write the object data to the CSV file
                writer.writerow([i, class_id, object_name, conf, *cords, distance])
                
                
        #################################################

        # DRAW BOUNDING BOXES

        #################################################

        for box in detected_objects:
            # Extract box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Get class ID and confidence
            class_id = box.cls[0].item()
            conf = box.conf[0].item()
            # Get object name
            object_name = object_names.get(class_id, 'Unknown')
            # Draw bounding box on the image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put text showing class name and confidence
            cv2.putText(frame, f'{object_name}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Put text showing distance
            cv2.putText(frame, f'Distance: {distance:.2f}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                
        #################################################

        # DRAW LINES

        #################################################
        if not isinstance(lines, list):
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        
        ################################################

        # Other

        ################################################   
        
        # Increment frame value +1
        i += 1
        
        # Write the frame to the .avi file
        out.write(frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    else:
        # Break the loop if the end of the video is reached
        break
        
################################################

# CLEAN-UP

################################################  

# Release the video capture object, writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()


