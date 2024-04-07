# Import necessary modules
import cv2
import csv
import argparse
from functions.distance_functions import DistanceEstimator
from functions.hough_functions import LineDetector
from functions.run_frame import FrameAnalyzer
from ultralytics import YOLO

def main(image_path):
    # Build YOLO, detector, and estimator
    yolo_model = YOLO('../models/yolov8n.pt')
    detector = LineDetector()
    estimator = DistanceEstimator()
    fa = FrameAnalyzer(yolo_model, detector, estimator)

    # Load the image
    image = cv2.imread(image_path)

    # Establish output file paths
    csv_file_name = '../output/image_objects.csv'
    image_output_path = '../output/processed_image.jpg'

    # Run the frame analyzer on the image
    analyzed_image = fa.run_frame(image, csv_file_name)

    # Write the altered image to the output path
    cv2.imwrite(image_output_path, analyzed_image)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Detect objects in an image and save results to a CSV file.')
    parser.add_argument('--image', '-i', type=str, default='inputs/straight_object.png',
                        help='Path to the input image file')
    args = parser.parse_args()

    # Call the main function with the specified image path
    main(args.image)



