import argparse
from importlib import reload
from ultralytics import YOLO
import numpy as np
import pandas as pd
import time
import csv
import cv2
import functions.hough_functions
import functions.other_functions

reload(functions.hough_functions)
reload(functions.other_functions)

from functions.other_functions import DistanceEstimator
from functions.hough_functions import LineDetector


parser = argparse.ArgumentParser(description='Detect objects in an image and save results to a CSV file.')
parser.add_argument('--image', '-i', type=str, default='inputs/straight_object.png', help='Path to the input image file')
args = parser.parse_args()

################################################

# BUILD / PULL MODEL

################################################
start = time.time()
model = YOLO('models/yolov8n.pt')
#results = model.train(data='coco128.yaml', epochs=3)
#results = model.val()
print(f"{np.around(time.time() - start, 4)*1000} milliseconds loading in YOLO")

################################################

# LOAD IMAGE AND APPLY MODELS

################################################

start = time.time()
# Load the image
image_path = args.image
image = cv2.imread(image_path)
print(f"{np.around(time.time() - start, 4)*1000} milliseconds loading in image")

start = time.time()
# Apply the YOLO object detection model
results = model(image_path)
print(f"{np.around(time.time() - start, 4)*1000} milliseconds detecting objects w/ YOLO")

start = time.time()
# Apply the Hough Line Transform
detector = LineDetector()
lines = detector.detect_lines_image(image_path)
print(f"{np.around(time.time() - start, 4)*1000} milliseconds detecting lines")

start = time.time()
# Calculate Distances
estimator = DistanceEstimator()
print(f"{np.around(time.time() - start, 4)*1000} milliseconds calculating distance")

################################################

# WRITE RESULTS TO CSV

################################################

start = time.time()
detected_objects = results[0].boxes
csv_file_name = 'output/image_objects.csv'
object_names = results[0].names
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['Class ID', 'Object Name','Confidence','Distance','X1', 'Y1', 'X2', 'Y2'])

    for box in detected_objects:
        class_id = box.cls[0].item()
        conf = box.conf[0].item()
        cords = box.xyxy[0].tolist()  # formatted as [x1, y1, x2, y2]
        object_name = object_names.get(class_id, 'Unknown')
        
        # Write the object data to the CSV file
<<<<<<< HEAD
        distance, direction = estimator.estimate_distance_direction(lines, image.shape[0], cords[1])
        writer.writerow([class_id, object_name, conf, distance, *cords])
=======
        distance, direction = estimator.estimate_distance_direction(lines, image.shape[0], cords[3])
        writer.writerow([class_id, object_name, conf, *cords, distance])
>>>>>>> b3fd3bba (Guys this works now!)

print(f"{np.around(time.time() - start, 4)*1000} milliseconds writing to csv")
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
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Put text showing class name and confidence
    cv2.putText(image, f'{object_name}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Put text showing distance
    cv2.putText(image, f'Distance: {distance:.2f}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#################################################

# DRAW LINES

#################################################

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#################################################

# WRITE TO LOCATION

#################################################

cv2.imwrite('output/processed_image.jpg', image)


