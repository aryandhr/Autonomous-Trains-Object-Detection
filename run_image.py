import argparse
from ultralytics import YOLO
import csv
import cv2

parser = argparse.ArgumentParser(description='Detect objects in an image and save results to a CSV file.')
parser.add_argument('--image', '-i', type=str, default='bus.jpg', help='Path to the input image file')
args = parser.parse_args()

################################################

# BUILD / PULL MODEL

################################################

model = YOLO('yolov8n.pt')
#results = model.train(data='coco128.yaml', epochs=3)
#results = model.val()

################################################

# LOAD IMAGE AND APPLY MODEL

################################################

image_path = args.image
image = cv2.imread(image_path)
results = model(image_path)

################################################

# WRITE RESULTS TO CSV

################################################

detected_objects = results[0].boxes
csv_file_name = 'detected_objects.csv'
object_names = results[0].names
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['Class ID', 'Object Name','Confidence', 'X1', 'Y1', 'X2', 'Y2'])

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

#################################################

# WRITE TO LOCATION

#################################################

cv2.imwrite('processed_image.jpg', image)


