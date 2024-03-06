import cv2
import math

class LineDetector:
    def __init__(self):
        pass

    def detect_lines_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print("Error opening image")
            return
        processed_image = self.process_frame(image)
        cv2.imshow('Result Image', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('hough_output_image.jpg', processed_image)

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
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame


# usage
detector = LineDetector()
image_path = 'Skylines_input_image.png'

detector.detect_lines_image(image_path)

