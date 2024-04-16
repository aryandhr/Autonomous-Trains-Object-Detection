"""
A class for analyzing frames by applying YOLO object detection and Hough Line Transform.

Attributes:
- yolo: YOLO object detection model.
- detector: Detector for applying Hough Line Transform.
- estimator: DistanceEstimator object for estimating distances.

Methods:
- __init__(self, yolo, detector, estimator): Initialize the FrameAnalyzer class.
- run_frame(self, frame, csv_path=None, n_frame=None): Analyze a frame, write detected objects to CSV,
  and return the analyzed frame.

Parameters:
- frame: Frame in image or video format to analyze.
- csv_path: Path to write CSV output (optional).
- n_frame: Frame number for logging (optional).

Returns:
- Altered analyzed frame.
"""

import csv
import cv2

class FrameAnalyzer:
    def __init__(self, yolo, detector, estimator):
        self.yolo = yolo
        self.detector = detector
        self.estimator = estimator

    def run_frame(self, frame, csv_path=None, n_frame = None):
        """
        Analyze a frame (image or video frame), write detected objects to CSV,
        and return the altered analyzed frame.
        
        :param frame: Frame in image or video format to analyze
        :param csv_path: Path to write CSV output (optional)
        :return: Altered analyzed frame
        """
        # Apply the YOLO object detection model
        results = self.yolo(frame)
        detected_objects = results[0].boxes
        object_names = results[0].names

        # Apply the Hough Line Transform
        lines = self.detector.detect_lines_frame(frame)
        self.estimator.analyze_lines(lines)

        object_distances = []

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            if csv_path is not None and n_frame is None:
                writer.writerow(['Trajectory','Class ID', 'Object Name',
                                 'Confidence', 'Distance', 'X1', 'Y1', 'X2', 'Y2'])
            elif csv_path is not None and n_frame == 0:
                writer.writerow(['Frame','Trajectory','Class ID', 'Object Name',
                                 'Confidence', 'Distance', 'X1', 'Y1', 'X2', 'Y2', ])

            # Loop through each detected object and write its information to CSV
            for box in detected_objects:
                class_id = box.cls[0].item()
                conf = box.conf[0].item()
                cords = box.xyxy[0].tolist()  # formatted as [x1, y1, x2, y2]
                object_name = object_names.get(class_id, 'Unknown')

                # Estimate distance using frame shape and object coordinates
                distance = self.estimator.estimate_distance(frame.shape[0], cords[3])
                object_distances.append(distance)

                if csv_path is not None and n_frame is None:
                    writer.writerow([self.estimator.trajectory, class_id, object_name,
                                     conf, *cords, distance])
                elif csv_path is not None:
                    writer.writerow([n_frame, self.estimator.trajectory, class_id, object_name,
                                     conf, *cords, distance])

        # Draw bounding boxes and display object information on the frame
        for idx, box in enumerate(detected_objects):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = box.cls[0].item()
            conf = box.conf[0].item()
            object_name = object_names.get(class_id, 'Unknown')

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display class name and confidence
            cv2.putText(frame, f'{object_name}: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display distance information
            if isinstance(object_distances[idx], (int, float)):
                distance = float(object_distances[idx])
                cv2.putText(frame, f'Distance: {distance:.2f}', (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f'Distance: Unknown', (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Draw lines detected by Hough Line Transform
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame
