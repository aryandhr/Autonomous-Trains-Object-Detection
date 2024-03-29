import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def slope_intercept(line):
    """
    :param line: Coordinates [[x1, y1, x2, y2]] that define a line.
    :return: [slope, intercept]
    """
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]


    slope = 0
    intercept = 0

    if y2-y1 != 0:
        slope = (x2 - x1) / (y2 - y1)
        intercept = x1 - slope * y1

    return [slope, intercept]

def calculate_x(y, slope, intercept):
    return y * slope + intercept

def calculate_distance(w0, w1, d0):
    '''
    :param w0: Width of the track in pixels at the edge of the screen (bottom)
    :param w1: Width of the track in pixels at the bottom of the detected object
    :param d1: Distance from the camera to w0 (in Feet)
    :return: Distance to w1
    '''
    return d0 * (w0/w1)

def height_width(image):
    '''
    :param image: An image imported through cv2.imread()
    :return: Two ints representing the height and width of the image in pixels respectively
    '''
    return image.shape[:2]

def k_cluster_lines(lines):
    """
    this is a function that will intake a vector of subvectors
    and then run k-means clustering on them to find the two lines
    that can be used to calculate the distance of an object
    :param lines: vector of subvectors containing [x1,y1,x2,y2]
    :return: slope and intercept for two values
    """
    X = pd.DataFrame(np.zeros(shape = (lines.shape[0], 2)))
    X.columns = ['slope','intercept']

    for i, line in enumerate(lines):
        X.iloc[i,:] = slope_intercept(line)

    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    X['group'] = kmeans.predict(X)

    return X


def find_line_equations(X):
    """
    meant to be used in conjunction with k_cluster_lines
    given dataframe of slopes, intercepts, and groups,
    identify the equations for the left and right rails
    :param X: dataframe returned from k_cluster_lines
    :return: left line slope and intercept, right line slope and intercept
    """
    for group in np.unique(X['group']):
        X_group = X[X['group'] == group]
        slope = X_group.mean(axis = 0)['slope']
        if slope < -0.05:
            left_line = X_group.mean(axis = 0).drop('group')
        elif slope > 0.05:
            right_line = X_group.mean(axis = 0).drop('group')

    return left_line, right_line


def find_x(slope, intercept, y):
    return y * slope + intercept


def distance_estimation(X, y0, y1, d0):
    """

    :param X: dataframe of X from find_line_equations(X)
    :param y0: yvalue at the bottom of the screen
    :param y1: yvalue at the bottom of the bounding box of an object
    :param d0: distance from camera to bottommost visible part of screen
    :return: estimation of distance from camera to object
    """
    left_line, right_line = find_line_equations(X)

    if left_line is None or right_line is None:
        return None

    widths = []
    for y in [y0, y1]:
        left_x = find_x(left_line.slope, left_line.intercept, y)
        right_x = find_x(right_line.slope, right_line.intercept, y)
        widths.append(right_x - left_x)

    return calculate_distance(widths[0], widths[1], d0)


def turn_guesstimation(left_slope, right_slope):
    """
    given slope of left and right line, determine direction of train tracks
    :param left_line: slope and intercept of left rail line
    :param right_line: slope and intercept of right rail line
    :return: guesstimation on turning left or right
    """
    if -left_slope / right_slope < .9:
        return "turning left"
    elif -left_slope / right_slope  > 1.1:
        return "turning right"
    else:
        return "straight away"




def putting_it_all_together(lines, y0, y1, d0):
    """

    :param lines: vector of subvectors with [x1,y1,x2,y2]
    :param y0: yvalue at the bottom of the screen
    :param y1: yvalue at the bottom of the bounding box of an object
    :param d0: distance from camera to bottommost visible part of screen
    :return:
    """

    X = k_cluster_lines(lines)
    left_line, right_line = find_line_equations(X)

    print(distance_estimation(X, y0, y1, d0))
    print(turn_guesstimation(left_line.slope, right_line.slope))

    return