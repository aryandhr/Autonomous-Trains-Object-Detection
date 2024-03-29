import cv2
import math
import numpy as np

class LineDetector:
    def __init__(self):
        pass

    def crop_image(self, image):
        '''
        This function crops the image so the result is only a small middle chunk. This makes it easier for the Hough Line
        transform to identify only the train tracks rather than random lines throughout the rest of the image.
        Crops the bottom half of the middle 1/5th of the image.
        
        :param image: An image imported through cv2.imread()
        :return: Cropped image according to rules specified within function.
        '''
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Define the region of interest (ROI)
        roi_x = 2 * width // 5  # Starting x-coordinate of the ROI (left boundary of the middle fifth)
        roi_width = width // 5  # Width of the ROI (1/5 of the width of the image)

        # Define the bottom 2/3 of the ROI
        bottom = height # The height of the image = the bottom coordinate
        bottom_roi_height = height//2  # Height of the bottom half

        # Crop the region of interest from the image
        roi = image[bottom_roi_height:bottom, roi_x:roi_x+roi_width]

        return roi, roi_x, bottom_roi_height
    
    def detect_lines_image(self, image_path, crop=True, output = False):
        '''
        This function is responsible for cropping an image if needed and then detecting the railroad tracks in an image.
        The goal is to return just 2 sets of coordinates representing the 2 lines that are the tracks.
        
        :param image_path: Path to image in your directory
        :param crop: Set to false if you want to detect lines in the whole image rather than subset for rails
        :return: Cropped image according to rules specified within function.
        '''
        # Read in the image
        image = cv2.imread(image_path)
        
        # Initialize x adjustment values to 0
        # These values will be adjust the cropped coordinate system to the overall picture coordinate system
        x_adj = 0
        y_adj = 0
        
        # If image is None then we gots sum problems bby
        if image is None:
            print("Error opening image")
            return
        # If we are cropping then we should crop
        elif crop:
            image, x_tmp, y_tmp = self.crop_image(image)
            # x_tmp as the code is currently written is the number of pixels that 2/5 of the image takes up horizontally
            x_adj += x_tmp
            # y_tmp is half the image height in pixels
            y_adj += y_tmp
            
        # process the image and look for lines
        processed_image, lines = self.process_frame(image)
        height, width, _ = image.shape
        for line in lines:
            # Manually adjust the x and y values to reset the coordinate system to the main photo rather than the crop
            # line = line[0]
            line[0] += x_adj
            line[1] += y_adj
            line[2] += x_adj
            line[3] += y_adj
            x1, y1, x2, y2 = line
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                # intercept = y1 - slope * x1
                if output:

                    print(f"Line Detected with equation x = (y - {y1}) / {slope} + {x1}")
            elif output:
                print(f"Vertical Line Detected at x = {x1}")
        #cv2.imwrite('hough_output_image.jpg', processed_image)

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
        
        ###########################
        
        # MODIFY minLineLength and maxLineGap to tweak for the video/image you are running on
        # This will adjust Hough so that it can detect either less lines, or more lines
        
        ###########################
        
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 100, minLineLength=200, maxLineGap=10)
        #if lines is not None:
        #    for line in lines:
        #        line = line[0]
        #        x1, y1, x2, y2 = line
        #        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        lines = np.squeeze(lines)

        return frame, lines

# '''
# usage
# detector = LineDetector()
# image_path = 'ana.png'
#
# lines = detector.detect_lines_image(image_path)
# print(lines)
# '''
