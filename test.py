print("Starting the test program...")

# Import necessary packages
print("Importing Packages...")
from ultralytics import YOLO
import csv

# Load a pretrained YOLO model (recommended for training)
print("Loading pre-trained model...")
model = YOLO('models/yolov8n.pt')

# Perform object detection on an image using the model
print("Performing Model Detection!...")
results = model('https://ultralytics.com/images/bus.jpg')

detected_objects = results[0].boxes
csv_file_name = 'output/detected_objects.csv'
object_names = results[0].names
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['Class ID', 'Object Name','Confidence', 'X1', 'Y1', 'X2', 'Y2'])
    
    # Loop through each detected object
    for box in detected_objects:
        class_id = box.cls[0].item()
        conf = box.conf[0].item()
        cords = box.xyxy[0].tolist()  # formatted as [x1, y1, x2, y2]
        object_name = object_names.get(class_id, 'Unknown')
        
        # Write the object data to the CSV file
        writer.writerow([class_id, object_name, conf, *cords])

print("Program Complete! Shutting down...")
