import cv2
import math

class LineDetector:
    def __init__(self):
        pass

#     def detect_lines_image(self, image_path):
#         image = cv2.imread(image_path)
#         if image is None:
#             print("Error opening image")
#             return
#         processed_image, lines = self.process_frame(image)
#         height, width, _ = image.shape
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             slope = (y2-y1)/(x2-x1)
#             print(f"Line Detected with equation y = {slope} * (x - {x1}) + {y1}")
#         #cv2.imshow('Result Image', processed_image)
#         #cv2.waitKey(0)
#         #cv2.destroyAllWindows()
#         cv2.imwrite('hough_output_image.jpg', processed_image)

    def detect_lines_image(self, image_path, path=True):
        if path is True:
            image = cv2.imread(image_path)
        else:
            image = image_path
        if image is None:
            print("Error opening image")
            return
        processed_image, lines = self.process_frame(image)
        height, width, _ = image.shape
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                print(f"Line Detected with equation x = (y - {y1}) / {slope} + {x1}")
            else:
                print(f"Vertical Line Detected at x = {x1}")
        # cv2.imshow('Result Image', processed_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('hough_output_image.jpg', processed_image)
        return lines

    def detect_lines_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('hough_output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                processed_frame = self.process_frame(frame)
                out.write(processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 100, minLineLength=200, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame, lines

'''
# usage
detector = LineDetector()
image_path = 'Skylines_input_image.png'

detector.detect_lines_image(image_path)
'''
