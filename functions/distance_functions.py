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
        self.left_line = [None,None]
        self.left_intercepts = deque(maxlen = deque_length)
        self.left_slopes = deque(maxlen=deque_length)

        self.right_line = [None,None]
        self.right_intercepts = deque(maxlen=deque_length)
        self.right_slopes = deque(maxlen=deque_length)

        self.trajectory = None

        self.kmeans = KMeans(n_clusters=2, n_init=10)

        # should be a *constant or relatively unchanging value
        self.d0 = dist_to_cam

    def analyze_lines(self,lines):
        """
        inputs lines, kcluster them, fit them to self object, guess on left or right turn
        :param lines: list of [x1,y1,x2,y2]
        :return: None
        """
        if len(lines) > 0:
            self.k_cluster_lines(lines)
            self.fit_line_equations()
            self.turn_guesstimation()


    def k_cluster_lines(self, lines):
        """
        this is a function that will intake a vector of subvectors
        and then run k-means clustering on them to find the two lines
        that can be used to calculate the distance of an object
        :param lines: vector of subvectors containing [x1,y1,x2,y2]
        :return: slope and intercept for two values
        """
        # creating empty dataframe to store intercepts and slopes in
        X = pd.DataFrame(np.zeros(shape = (lines.shape[0], 3)))
        X.columns = ['intercept','slope','group']

        # iterate through each x1, y1, x2, y2 and calculate slope and interecept
        for i, line in enumerate(lines):
            X.iloc[i,:2] = self.slope_intercept(line)
            X.iloc[i,2] = 1

        # perform unsupervised kmeans clustering (n=2) on the dataframe
        if X.shape[0] > 1:
            self.kmeans.fit(X)
            # predict and save the output to X
            X['group'] = self.kmeans.predict(X)

        left_group = pd.DataFrame(np.zeros(shape=(0, 2)), columns=['intercept', 'slope'])
        right_group = pd.DataFrame(np.zeros(shape=(0, 2)), columns=['intercept', 'slope'])

        # loop through each group
        for group in np.unique(X['group']):
            X_group = X[X['group'] == group]
            slope = X_group['slope'].mean()

            # identify if group is the left line by having a negative slope
            if slope < -0.05:
                left_group = pd.concat([left_group, X_group[['intercept', 'slope']]], ignore_index=True)
            # positive slope = right line
            elif slope > 0.05:
                right_group = pd.concat([right_group, X_group[['intercept', 'slope']]], ignore_index=True)

        # failsafe
        self.left_line = [None, None]
        self.right_line = [None, None]

        if left_group.shape[0] > 0:
            left_line = left_group.mean(axis=0)
            self.left_line = [left_line['intercept'], left_line['slope']]

        if right_group.shape[0] > 0:
            right_line = right_group.mean(axis=0)
            self.right_line = [right_line['intercept'], right_line['slope']]


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


    def fit_line_equations(self):
        """

        :param left_line:
        :param right_line:
        :return:
        """


        # if lines were detected, append them to the lines of previous frames
        if None not in self.left_line:
            self.left_intercepts.append(self.left_line[0])
            self.left_slopes.append(self.left_line[1])

        # set line values equal to the mean values of the previous 10 frames
        if len(self.left_intercepts) > 0:
            self.left_line[0] = sum(self.left_intercepts) / len(self.left_intercepts)
            self.left_line[1] = sum(self.left_slopes) / len(self.left_intercepts)

        if None not in self.right_line:
            self.right_intercepts.append(self.right_line[0])
            self.right_slopes.append(self.right_line[1])

        if len(self.right_intercepts) > 0:
            self.right_line[0] = sum(self.right_intercepts) / len(self.right_intercepts)
            self.right_line[1] = sum(self.right_slopes) / len(self.right_intercepts)



    def estimate_distance(self, y0, y1):
        """
        :param y0: yvalue at the bottom of the screen
        :param y1: yvalue at the bottom of the bounding box of an object
        :return: estimation of distance from camera to object
        """

        # only calculate if left_line and right_line have information
        if None not in self.left_line and None not in self.right_line:
            widths = []
            for y in [y0, y1]:
                # calculate the width difference between the right and left line
                left_x = self.calculate_x(self.left_line, y)
                right_x = self.calculate_x(self.right_line, y)
                widths.append(right_x - left_x)

            return self.calculate_distance(widths[0], widths[1])

        else:
            return "Error: No Lines Detected"

    def calculate_x(self, line, y):
        return line[0] + line[1] * y


    def calculate_distance(self, w0, w1):
        '''
        :param w0: Width of the track in pixels at the edge of the screen (bottom)
        :param w1: Width of the track in pixels at the bottom of the detected object
        :return: Distance d1 at w1

        this is a fun thing with trig, called the law of similar triangles
        because we know the train tracks have the same width down the line in reality
        though in the image their lines aren't parallel due to perspective
        we can compare the widths of the tracks at two points, and by knowing
        the distance of one of those points, we can find the distance of the other

        because the visible part of the train tracks at the bottom of the frame will *likely
        always stay constant, we can estimate the distance d1
        '''

        return max(0,self.d0 * (w0/w1))



    def turn_guesstimation(self):
        """
        given slope of left and right line, determine direction of train tracks
        :return: left turn, right turn, straight track, or error no lines detected

        by analyzing the slopes in the left and right lines, we can estimate if we're
        turning left, right, or going straight by checking which slope is steeper

        """
        # this means that the right slope is more angled (x = intercept + slope * y) so it's turning left
        if None not in self.left_line and None not in self.right_line:
            self.trajectory = "Straight Track"
            ratio = -1 * self.left_line[1] / self.right_line[1]
            if ratio < .8:
                self.trajectory = "Left Curve"
            elif ratio > 1.2:
                self.trajectory = "Right Curve"
            return self.trajectory
        return "Error: No Lines Detected"


