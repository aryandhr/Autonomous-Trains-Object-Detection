print("Starting the test program...")

# Import necessary packages
print("Importing Packages...")
from ultralytics import YOLO
import csv
import cv2

# Load a pretrained YOLO model (recommended for training)
print("Loading pre-trained model...")
model = YOLO('models/yolov8n.pt')

# Perform object detection on an image using the model
print("Performing Model Detection!...")
image_path = 'inputs/bus.jpg'
image = cv2.imread(image_path)

# Apply the model
results = model(image)

detected_objects = results[0].boxes
csv_file_name = 'output/test_objects.csv'
object_names = results[0].names

################################################

# WRITE RESULTS TO CSV

################################################

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

cv2.imwrite('output/test_image.jpg', image)

print("Program Complete! Shutting down...")
