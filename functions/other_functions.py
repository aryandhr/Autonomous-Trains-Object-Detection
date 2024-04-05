import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import deque

class DistanceEstimator:

    def __init__(self, deque_length = 10, dist_to_cam = 9.5):
        """

        :param deque_length: How many previous line calculations do we want to be taking the average of when calculating
            distance

        :param dist_to_cam:
             this is the distance from the camera to the bottom of the frame where the train tracks become visible
        """
        # information for storing slopes and intercepts of previous frames
        self.left_intercepts = deque(maxlen = deque_length)
        self.left_slopes = deque(maxlen=deque_length)
        self.right_intercepts = deque(maxlen=deque_length)
        self.right_slopes = deque(maxlen=deque_length)

        # should be a *constant or relatively unchanging value
        self.d0 = dist_to_cam


    def estimate_distance_direction(self, lines, y0, y1):
        """

        :param lines: vector of subvectors with [x1,y1,x2,y2]
        :param y0: yvalue at the bottom of the screen
        :param y1: yvalue at the bottom of the bounding box of an object
        :param d0: distance from camera to bottommost visible part of screen
        :return:
        """
        # get useable slopes and intercepts for the lines
        left_line, right_line = self.k_cluster_lines(lines)
        left_line, right_line = self.find_line_equations(left_line, right_line)
        # perform distance and direction estimation on the lines
        distance = self.distance_estimation(left_line, right_line, y0, y1)
        direction = self.turn_guesstimation(left_line[1], right_line[1])

        return distance, direction


    def k_cluster_lines(self, lines):
        """
        this is a function that will intake a vector of subvectors
        and then run k-means clustering on them to find the two lines
        that can be used to calculate the distance of an object
        :param lines: vector of subvectors containing [x1,y1,x2,y2]
        :return: slope and intercept for two values
        """
        # creating empty dataframe to store intercepts and slopes in
        X = pd.DataFrame(np.zeros(shape = (lines.shape[0], 2)))
        X.columns = ['intercept','slope']

        # iterate through each x1, y1, x2, y2 and calculate slope and interecept
        for i, line in enumerate(lines):
            X.iloc[i,:] = self.slope_intercept(line)

        # perform unsupervised kmeans clustering (n=2) on the dataframe
        kmeans = KMeans(n_clusters = 2, n_init = 10)
        kmeans.fit(X)

        # predict and save the output to X
        X['group'] = kmeans.predict(X)

        # failsafe
        left_line = None
        right_line = None

        # loop through each group
        for group in np.unique(X['group']):
            X_group = X[X['group'] == group]
            slope = X_group.mean(axis = 0)['slope']
            # identify if group is the left line by having a negative slope
            if slope < -0.05:
                left_line = X_group.mean(axis = 0).drop('group')
                left_line = [left_line['intercept'], left_line['slope']]
            # positive slope = right line
            elif slope > 0.05:
                right_line = X_group.mean(axis = 0).drop('group')
                right_line = [right_line['intercept'], right_line['slope']]

        return left_line, right_line

    def slope_intercept(self, line):
        """
        :param line: Coordinates [[x1, y1, x2, y2]] that define a line.
        :return: [slope, intercept]
        """
        # get coordinates from line
        x1, y1, x2, y2 = line[0]

        slope = 0
        intercept = 0

        # calculate slope and intercept in terms of x = intercept + slope * y
        if y2-y1 != 0:
            slope = (x2 - x1) / (y2 - y1)
            intercept = x1 - slope * y1

        return intercept, slope


    def find_line_equations(self, left_line, right_line):
        """
        meant to be used in conjunction with k_cluster_lines
        given dataframe of slopes, intercepts, and groups,
        identify the equations for the left and right rails
        :param X: dataframe returned from k_cluster_lines
        :return: left line slope and intercept, right line slope and intercept
        """

        if left_line is None:
            left_line = []

        # if lines were detected, append them to the lines of previous frames
        else:
            self.left_intercepts.append(left_line[0])
            self.left_slopes.append(left_line[1])

        # set line values equal to the mean values of the previous 10 frames
        if len(self.left_intercepts) > 0:
            left_line[0] = sum(self.left_intercepts) / len(self.left_intercepts)
            left_line[1] = sum(self.left_slopes) / len(self.left_intercepts)

        if right_line is None:
            right_line =[]
        else:
            self.right_intercepts.append(right_line[0])
            self.right_slopes.append(right_line[1])

        if len(self.right_intercepts) > 0:
            right_line[0] = sum(self.right_intercepts) / len(self.right_intercepts)
            right_line[1] = sum(self.right_slopes) / len(self.right_intercepts)

        return left_line, right_line



    def distance_estimation(self, left_line, right_line, y0, y1):
        """

        :param left_line: [intercept, slope] of left rail
        :param right_line: [intercept, slope] of right rail
        :param y0: yvalue at the bottom of the screen
        :param y1: yvalue at the bottom of the bounding box of an object
        :param d0: distance from camera to bottommost visible part of screen
        :return: estimation of distance from camera to object
        """

        # only calculate if left_line and right_line have information
        if len(left_line) > 0 and len(right_line) > 0:
            widths = []
            for y in [y0, y1]:
                # calculate the width difference between the right and left line
                left_x = self.calculate_x(left_line[0], left_line[1], y)
                right_x = self.calculate_x(right_line[0], right_line[1], y)
                widths.append(right_x - left_x)

            return self.calculate_distance(widths[0], widths[1])

        else:
            return None

    def calculate_x(self, intercept, slope, y):
        return intercept + slope * y


    def calculate_distance(self, w0, w1):
        '''
        :param w0: Width of the track in pixels at the edge of the screen (bottom)
        :param w1: Width of the track in pixels at the bottom of the detected object
        :param d1: Distance from the camera to w0 (in Feet)
        :return: Distance d1 at w1


        this is a fun thing with trig, called the law of similar triangles
        because we know the train tracks have the same width down the line in reality
        though in the image their lines aren't parallel due to perspective
        we can compare the widths of the tracks at two points, and by knowing
        the distance of one of those points, we can find the distance of the other

        because the visible part of the train tracks at the bottom of the frame will *likely
        always stay constant, we can estimate the distance d1
        '''

        return self.d0 * (w0/w1)



    def turn_guesstimation(self, left_slope, right_slope):
        """
        given slope of left and right line, determine direction of train tracks
        :param left_line: slope and intercept of left rail line
        :param right_line: slope and intercept of right rail line
        :return: guesstimation on turning left or right

        by analyzing the slopes in the left and right lines, we can estimate
        if we're turning left, right, or going straight by checking which slope is steeper

        """
        # this means that the right slope is more angled (x = intercept + slope * y) so it's turning left
        if -left_slope / right_slope < .9:
            return "turning left"
        elif -left_slope / right_slope > 1.1:
            return "turning right"
        return "straight away"


