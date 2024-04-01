############################
#
# james
#    first tests with YOLOv8
#  intramotev team
#
#
#############################

# make sure ultralytics, cv2 are installed using pip



# imports
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import csv

# pretrained model
model = YOLO("yolov8n.pt")

# open video file
video_1 = "cars_1.mp4"
cap = cv2.VideoCapture(video_1)

# Prep the CSV file
csv_file_name = "detected_objects.csv"
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame','Class ID', 'Object Name','Confidence', 'X1', 'Y1', 'X2', 'Y2'])

# Define the codec and create VideoWriter object
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize a list to store detected objects
detected_objects = []
i = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        ################################
        
        ### JOES CODE TO WRITE TO FILE
        
        ################################
        
        detected_objects = results[0].boxes
        csv_file_name = 'detected_objects.csv'
        object_names = results[0].names
        with open(csv_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            for box in detected_objects:
                class_id = box.cls[0].item()
                conf = box.conf[0].item()
                cords = box.xyxy[0].tolist()  # formatted as [x1, y1, x2, y2]
                object_name = object_names.get(class_id, 'Unknown')

                # Write the object data to the CSV file
                writer.writerow([i, class_id, object_name, conf, *cords])

                        
        ################################
        
        ### VISUALIZE STUFF
        
        ################################
        
        '''
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        '''
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()

