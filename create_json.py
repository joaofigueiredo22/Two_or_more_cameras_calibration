#!/usr/bin/env python2
"""
This code shows how the optimizer class may be used for changing the colors of a set of images so that the average
color in all images is very similar. This is often called color correction.
The OCDatasetLoader is used to collect data from a OpenConstructor dataset
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import argparse  # to read command line arguments
import math

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import cv2
import KeyPressManager.KeyPressManager
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import json
import itertools


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

def generate_chessboard(size, dimensions):
    objp = np.zeros((dimensions[0] * dimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2)
    objp = objp * size
    return objp

def find_cam_chess_realpoints(fname, k_matrix, dist_matrix):
    objpoints = []
    imgpoints = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = cv2.imread(fname)

    if img is None:
        raise ValueError('Could not read image from ' + str(fname))
    # print(img.shape)
    # cv2.imshow('gui', img)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(pts_chessboard)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
    else:
        return 0

    return 1


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------

    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    ap.add_argument("-d", "--dataset_path", help="Path to the dataset", type=str, required=True)
    ap.add_argument("-j", "--json_path", help="Full path to Json file", type=str, required=True)
    args = vars(ap.parse_args())

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------

    json_file = args['json_path']

    # Image used
    if not args['dataset_path'][-1] == '/':  # make sure the path is correct
        args['dataset_path'] += '/'

    f = open(args['json_path'], 'r')
    calibration_data = json.load(f)

    dimensions = calibration_data['calibration_config']['calibration_pattern']['dimension']
    size_board = calibration_data['calibration_config']['calibration_pattern']['size']
    pts_chessboard = generate_chessboard(size_board, dimensions)
    sensors = calibration_data['sensors']

    name_image_list = []
    k_matrix = np.zeros((3, 3, len(sensors)), np.float32)
    dist_matrix = np.zeros((len(sensors), 5), np.float32)
    i2 = 0
    sensor = []
    for i in calibration_data['collections']:
        i3 = 0
        for i1 in sensors:
            name_image = args['dataset_path'] + calibration_data['collections'][str(i)]['data'][str(i1)]['data_file']
            name_image_list.append(name_image)

            if i == "0":
                sensor.append(str(i1))
                k_matrix[i2, 0, :] = calibration_data['sensors'][str(i1)]['camera_info']['K'][0:3]
                k_matrix[i2, 1, :] = calibration_data['sensors'][str(i1)]['camera_info']['K'][3:6]
                k_matrix[i2, 2, :] = calibration_data['sensors'][str(i1)]['camera_info']['K'][6:9]
                dist_matrix[i2, :] = calibration_data['sensors'][str(i1)]['camera_info']['D']
                i2 += 1

            calibration_data['collections'][str(i)]['data'][str(i1)]['detected'] = find_cam_chess_realpoints(name_image, k_matrix[i3, :, :], dist_matrix[i3, :])
            i3 += 1

    with open('test1.txt', 'w') as outfile:
        json.dump(calibration_data, outfile)
    exit(0)
    img1 = cv2.imread(name_image[0])
    # img = cv2.drawChessboardCorners(img1, (9, 6), calibration.right_cam_image_points[ind1][0], True)
    cv2.imshow('img', img1)
    cv2.waitKey(0)