"""
File: run_live_video.py

Description: This script implements a real-time video processing pipeline to detect objects, estimate distances, and detect lines using a live camera feed. It uses the YOLO (You Only Look Once) model for object detection, integrates custom line detection via Hough Transform, and estimates distances to detected objects.

Requirements:
- OpenCV: For capturing and processing video frames from a live camera feed.
- Ultralytics YOLO: For real-time object detection.
- Custom Modules:
  - distance_functions: Contains the DistanceEstimator class for estimating distances to detected objects.
  - hough_functions: Contains the LineDetector class for line detection using Hough Transform.
  - run_frame: Contains the FrameAnalyzer class that integrates object detection, line detection, and distance estimation.

Usage:
The script continuously captures video frames from the default or specified camera, processes each frame to detect objects, estimate their distances, and detect lines, and displays the processed video in real time. The results are optionally saved in a CSV file and an output video file.

Output:
- A CSV file ('output/csvs/live_video_objects.csv') containing details of the detected objects for each frame.
- An output video ('output/videos_images/output_live_video.avi') showing the detected objects and lines for each frame (optional).

Controls:
- Press 'q' to quit the live video feed and close the application.
"""

import cv2
import csv
from functions.distance_functions import DistanceEstimator
from functions.hough_functions import LineDetector
from functions.run_frame import FrameAnalyzer
from ultralytics import YOLO

def main():
    # Build YOLO, detector, and estimator
    yolo_model = YOLO('models/yolov8n.pt')
    detector = LineDetector()
    estimator = DistanceEstimator()
    fa = FrameAnalyzer(yolo_model, detector, estimator)

    # Open camera device (default camera or specify a camera index)
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    # Define output CSV file
    csv_file_name = "output/csvs/live_video_objects.csv"
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Class ID', 'Object Name', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 'Distance'])

    # Set up a video writer for output (optional)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output/output_live_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    frame_idx = 0  # Frame index

    # Process each frame from the camera feed
    while True:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            print("Error: Failed to capture frame from camera")
            break

        # Analyze the frame using FrameAnalyzer
        analyzed_frame = fa.run_frame(frame, csv_file_name, frame_idx)

        # Display the analyzed frame (optional)
        cv2.imshow('Live Video Feed', analyzed_frame)

        # Write the analyzed frame to the output video (optional)
        if out is not None:
            out.write(analyzed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1  # Increment frame index

    # Clean up
    cap.release()  # Release the camera
    if out is not None:
        out.release()  # Release the output video writer
    cv2.destroyAllWindows()  # Close any open windows

if __name__ == '__main__':
    main()
