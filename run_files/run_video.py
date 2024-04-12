"""

File: run_video.py

Description: This script provides a comprehensive pipeline for detecting objects within a video, detecting lines using Hough Transform, and estimating distances to those objects. It processes video frames to identify and annotate objects, estimate their distance, and detect significant lines in each frame.

Requirements:
- OpenCV: For video processing and manipulation.
- Ultralytics YOLO: For object detection tasks.
- Custom Modules:
  - distance_functions: Contains the DistanceEstimator class for estimating distances to detected objects.
  - hough_functions: Contains the LineDetector class for line detection using Hough Transform.
  - run_frame: Contains the FrameAnalyzer class that integrates object detection, line detection, and distance estimation.

Usage:
The script accepts a video file path as an input argument.

Command Line Arguments:
- --video, -v: Path to the input video file. Default is 'inputs/train_clip.mp4'.

Output:
- A CSV file ('output/csvs/video_objects.csv') containing details of the detected objects for each frame.
- An output video ('output/videos_images/output_video.avi') showing the detected objects and lines for each frame.
"""

# Import necessary modules
import cv2
import csv
import argparse
from functions.distance_functions import DistanceEstimator
from functions.hough_functions import LineDetector
from functions.run_frame import FrameAnalyzer
from ultralytics import YOLO

def main(video_path):
    # Build YOLO, detector, and estimator
    yolo_model = YOLO('models/yolov8n.pt')
    detector = LineDetector()
    estimator = DistanceEstimator()
    fa = FrameAnalyzer(yolo_model, detector, estimator)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Define output CSV file
    csv_file_name = "output/csvs/video_objects.csv"
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Class ID', 'Object Name', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 'Distance'])

    # Define output video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output/videos_images/output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    frame_idx = 0  # Frame index

    # Process each frame in the video
    while cap.isOpened():
        success, frame = cap.read()  # Read a frame from the video

        if success:
            # Analyze the frame using FrameAnalyzer
            analyzed_frame = fa.run_frame(frame, csv_file_name, n_frame = frame_idx)
            out.write(analyzed_frame)  # Write the processed frame to the output video

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1  # Increment frame index

        else:
            break  # Break the loop if the end of the video is reached

    # Clean up
    cap.release()  # Release the video capture object
    out.release()  # Release the output video writer
    cv2.destroyAllWindows()  # Close any open windows

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect objects in a video and save results to a CSV file.')
    parser.add_argument('--video', '-v', type=str, default='inputs/train_clip.mp4', help='Path to the input video file')
    args = parser.parse_args()

    # Run the main function with the specified video path
    main(args.video)
