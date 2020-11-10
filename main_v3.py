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


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
from OptimizationUtils import utilities


class Calibration:

    def __init__(self):
        self.right_cam_rotation_vector = [0.0, 0.0, 0.0]
        self.center_cam_rotation_vector = [0.0, 0.0, 0.0]
        self.left_cam_rotation_vector = [0.0, 0.0, 0.0]
        self.right_cam_translation = [0.0, 0.0, 0.0]
        self.center_cam_translation = [0.0, 0.0, 0.0]
        self.left_cam_translation = [0.0, 0.0, 0.0]
        self.chess_rotation_vector = []
        self.chess_translation = []
        self.right_cam_image_points = []
        self.center_cam_image_points = []
        self.left_cam_image_points = []


def generate_chessboard(size, dimensions):
    objp = np.zeros((dimensions[0] * dimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2)
    objp = objp * size
    return objp


def find_cam_chess_realpoints(fname, left_or_right):
    objpoints_left = []
    imgpoints_left = []
    objpoints_center = []
    imgpoints_center = []
    objpoints_right = []
    imgpoints_right = []
    imgpoints_center = []
    # print(fname)
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
    if ret == True:
        if left_or_right == 0:
            objpoints_left.append(pts_chessboard)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners2)
            retval, rvec, tvec = cv2.solvePnP(objpoints_left[0], imgpoints_left[0], k_left,
                                              dist_left)  # calculating the rotation and translation vectors from left camera to chess board
        elif left_or_right == 1:
            objpoints_right.append(pts_chessboard)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_right.append(corners2)
            retval, rvec, tvec = cv2.solvePnP(objpoints_right[0], imgpoints_right[0], k_right,
                                              dist_right)  # calculating the rotation and translation vectors from left camera to chess board
        elif left_or_right == 2:
            objpoints_center.append(pts_chessboard)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_center.append(corners2)
            retval, rvec, tvec = cv2.solvePnP(objpoints_center[0], imgpoints_center[0], k_center,
                                              dist_center)  # calculating the rotation and translation vectors from left camera to chess board

    return rvec, tvec

def inv_tranformation_matrix(r, t):
    x_array = np.array(r[0])
    r_inv = r.T
    t_inv = np.zeros((3, 1), np.float32)
    t_inv[0] = -np.dot(r[0:3, 0], t)
    t_inv[1] = -np.dot(r[0:3, 1], t)
    t_inv[2] = -np.dot(r[0:3, 2], t)
    #
    # r_rot = []
    #
    # r_rot[0] = r[0].T * -1
    # t_rot = np.matmul(r_rot, t[0][0:3])
    # r[0] = r[0].T
    # t[0][0:3] = t_rot
    return r_inv, t_inv

if __name__ == "__main__":

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------

    ap = argparse.ArgumentParser()
    ap = OptimizationUtils.addArguments(ap)  # OptimizationUtils arguments
    ap.add_argument("-d", "--dataset_path", help="Path to the dataset", type=str, required=True)
    ap.add_argument("-j", "--json_path", help="Full path to Json file", type=str, required=True)
    ap.add_argument("-r", "--3d_points", help="Residuals based on 3D points", type=str, required=False)
    args = vars(ap.parse_args())

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    calibration = Calibration()

    json_file = args['json_path']

    f = open(args['json_path'], 'r')
    calibration_data = json.load(f)

    # Read from args type of residuals

    residuals_3d = 0
    if args['3d_points']:
        residuals_3d = 1
    # print(residuals_3d)
    # exit(0)
    # Chessboard dimensions
    dimensions = calibration_data['calibration_config']['calibration_pattern']['dimension']
    size_board = calibration_data['calibration_config']['calibration_pattern']['size']
    number_sensors = len(calibration_data['sensors'])
    sensors_list = calibration_data['sensors']
    print (number_sensors)
    number_sensors_list = np.zeros((number_sensors), np.int)
    for idx in range(number_sensors):
        number_sensors_list[idx] = idx
    sensors_combination_2 = list(itertools.combinations(sensors_list, 2))
    sensors_combination_3 = list(itertools.combinations(sensors_list, 3))

    ancored_sensor = sensors_combination_2[0][0]

    sensors_combination = []
    sensors_combination.extend(sensors_combination_2)
    sensors_combination.extend(sensors_combination_3)

    # K matrix and distortion coefficients from cameras
    k_left = np.zeros((3, 3), np.float32)
    k_center = np.zeros((3, 3), np.float32)
    k_right = np.zeros((3, 3), np.float32)

    k_left[0, :] = calibration_data['sensors']['top_left_camera']['camera_info']['K'][0:3]
    k_left[1, :] = calibration_data['sensors']['top_left_camera']['camera_info']['K'][3:6]
    k_left[2, :] = calibration_data['sensors']['top_left_camera']['camera_info']['K'][6:9]

    k_center[0, :] = calibration_data['sensors']['top_center_rgbd_camera_rgb']['camera_info']['K'][0:3]
    k_center[1, :] = calibration_data['sensors']['top_center_rgbd_camera_rgb']['camera_info']['K'][3:6]
    k_center[2, :] = calibration_data['sensors']['top_center_rgbd_camera_rgb']['camera_info']['K'][6:9]

    k_right[0, :] = calibration_data['sensors']['top_right_camera']['camera_info']['K'][0:3]
    k_right[1, :] = calibration_data['sensors']['top_right_camera']['camera_info']['K'][3:6]
    k_right[2, :] = calibration_data['sensors']['top_right_camera']['camera_info']['K'][6:9]

    dist_left = np.zeros((1, 5), np.float32)
    dist_center = np.zeros((1, 5), np.float32)
    dist_right = np.zeros((1, 5), np.float32)
    dist_left[:] = calibration_data['sensors']['top_left_camera']['camera_info']['D']
    # dist_center[:] = calibration_data['sensors']['top_center_rgbd_camera_rgb']['camera_info']['D']
    dist_right[:] = calibration_data['sensors']['top_right_camera']['camera_info']['D']

    # Image used
    if not args['dataset_path'][-1] == '/':  # make sure the path is correct
        args['dataset_path'] += '/'

    name_image_left = []
    name_image_right = []
    name_image_center = []
    for i in calibration_data['collections']:
    # for i in range(4):
        name_image_left.append(args['dataset_path'] + calibration_data['collections'][str(i)]['data']['top_left_camera']['data_file'])
        name_image_right.append(args['dataset_path'] + calibration_data['collections'][str(i)]['data']['top_right_camera']['data_file'])
        name_image_center.append(args['dataset_path'] + calibration_data['collections'][str(i)]['data']['top_center_rgbd_camera_rgb']['data_file'])
        calibration.chess_rotation_vector.append([0.0, 0.0, 0.0])
        calibration.chess_translation.append([0.0, 0.0, 0.0])


        # name_image_right.append(args['dataset_path'] + 'top_right_camera_' + str(i) + '.jpg')


    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating chessboard points
    pts_chessboard = generate_chessboard(size_board, dimensions)
    points_chessboard_homogeneous = np.zeros((4, dimensions[0] * dimensions[1]), np.float32)
    points_chessboard_homogeneous[0:3, :] = pts_chessboard.T
    points_chessboard_homogeneous[3, :] = 1


    # -------------------------------------------
    # ----- First Guess
    # -------------------------------------------
    rvec_cam_right, tvec_cam_right = find_cam_chess_realpoints(str(name_image_right[0]), 1)
    right_cam_T_chess_ground_truth = np.zeros((4, 4), np.float32)
    right_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_right)[0]
    right_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_right.T
    right_cam_T_chess_ground_truth[3, :] = [0, 0, 0, 1]  # homogenize


    rvec_cam_center, tvec_cam_center = find_cam_chess_realpoints(str(name_image_center[0]), 2)
    center_cam_T_chess_ground_truth = np.zeros((4, 4), np.float32)
    center_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_center)[0]
    center_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_center.T
    center_cam_T_chess_ground_truth[3, :] = [0, 0, 0, 1]  # homogenize

    rvec_cam_left, tvec_cam_left = find_cam_chess_realpoints(str(name_image_left[0]), 0)
    left_cam_T_chess_ground_truth = np.zeros((4, 4), np.float32)
    left_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_left)[0]
    left_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_left.T
    left_cam_T_chess_ground_truth[3, :] = [0, 0, 0, 1]  # homogenize

    # print(left_cam_T_chess_ground_truth)
    # print(center_cam_T_chess_ground_truth)
    # print(right_cam_T_chess_ground_truth)
    # exit(0)

    for i3 in range(len(name_image_left)):
        rvec_cam_left, tvec_cam_left = find_cam_chess_realpoints(str(name_image_left[i3]), 0)
        chess_ground_truth = np.zeros((4, 4), np.float32)
        chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_left)[0]
        chess_ground_truth[0:3, 3] = tvec_cam_left.T
        chess_ground_truth[3, :] = [0, 0, 0, 1]  # homogenize
        chess_first_guess_rotation, chess_first_guess_translation = inv_tranformation_matrix(chess_ground_truth[0:3, 0:3],
                                                                                             chess_ground_truth[0:3, 3])
        inv_chess_first_guess = inv_tranformation_matrix(chess_first_guess_rotation, chess_first_guess_translation)

        calibration.chess_rotation_vector[i3][0] = cv2.Rodrigues(chess_first_guess_rotation)[0][0]
        calibration.chess_rotation_vector[i3][1] = cv2.Rodrigues(chess_first_guess_rotation)[0][1]
        calibration.chess_rotation_vector[i3][2] = cv2.Rodrigues(chess_first_guess_rotation)[0][2]

        calibration.chess_translation[i3][0] = chess_first_guess_translation.T[0][0]
        calibration.chess_translation[i3][1] = chess_first_guess_translation.T[0][1]
        calibration.chess_translation[i3][2] = chess_first_guess_translation.T[0][2]

        if i3 == 0:
            chess_first_guess = np.zeros((4, 4), np.float32)
            chess_first_guess[0:3, 0:3] = chess_first_guess_rotation
            chess_first_guess[0:3, 3] = chess_first_guess_translation.T
            chess_first_guess[3, :] = [0, 0, 0, 1]  # homogenize

            right_cam_first_guess = np.matmul(right_cam_T_chess_ground_truth, chess_first_guess)
            center_cam_first_guess = np.matmul(center_cam_T_chess_ground_truth, chess_first_guess)

            calibration.right_cam_rotation_vector[0] = cv2.Rodrigues(right_cam_first_guess[0:3, 0:3])[0][0]
            calibration.right_cam_rotation_vector[1] = cv2.Rodrigues(right_cam_first_guess[0:3, 0:3])[0][1]
            calibration.right_cam_rotation_vector[2] = cv2.Rodrigues(right_cam_first_guess[0:3, 0:3])[0][2]

            calibration.right_cam_translation[0] = right_cam_first_guess[0, 3]
            calibration.right_cam_translation[1] = right_cam_first_guess[1, 3]
            calibration.right_cam_translation[2] = right_cam_first_guess[2, 3]

            calibration.center_cam_rotation_vector[0] = cv2.Rodrigues(center_cam_first_guess[0:3, 0:3])[0][0]
            calibration.center_cam_rotation_vector[1] = cv2.Rodrigues(center_cam_first_guess[0:3, 0:3])[0][1]
            calibration.center_cam_rotation_vector[2] = cv2.Rodrigues(center_cam_first_guess[0:3, 0:3])[0][2]

            calibration.center_cam_translation[0] = center_cam_first_guess[0, 3]
            calibration.center_cam_translation[1] = center_cam_first_guess[1, 3]
            calibration.center_cam_translation[2] = center_cam_first_guess[2, 3]

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('calibration', calibration)



    print("calibration model\n")
    print(calibration.right_cam_rotation_vector[0])
    print(opt.data_models['calibration'].right_cam_rotation_vector[0])

    # Create specialized getter and setter functions
    def setter(calibration, value, idx):
        if idx == 0:
            print("value\n")
            print(value)
            calibration.right_cam_rotation_vector[0] = value
        elif idx == 1:
            calibration.right_cam_rotation_vector[1] = value
        elif idx == 2:
            calibration.right_cam_rotation_vector[2] = value
        elif idx == 3:
            calibration.right_cam_translation[0] = value
        elif idx == 4:
            calibration.right_cam_translation[1] = value
        elif idx == 5:
            calibration.right_cam_translation[2] = value
        elif idx == 6:
            calibration.center_cam_rotation_vector[0] = value
        elif idx == 7:
            calibration.center_cam_rotation_vector[1] = value
        elif idx == 8:
            calibration.center_cam_rotation_vector[2] = value
        elif idx == 9:
            calibration.center_cam_translation[0] = value
        elif idx == 10:
            calibration.center_cam_translation[1] = value
        elif idx == 11:
            calibration.center_cam_translation[2] = value
        elif idx == 12:
            calibration.left_cam_rotation_vector[0] = value
        elif idx == 13:
            calibration.left_cam_rotation_vector[1] = value
        elif idx == 14:
            calibration.left_cam_rotation_vector[2] = value
        elif idx == 15:
            calibration.left_cam_translation[0] = value
        elif idx == 16:
            calibration.left_cam_translation[1] = value
        elif idx == 17:
            calibration.left_cam_translation[2] = value
        elif idx > 17:
            for ind in range(len(name_image_left)):
                if idx == 18 + 6 * ind:
                    calibration.chess_rotation_vector[ind][0] = value
                elif idx == 19 + 6 * ind:
                    calibration.chess_rotation_vector[ind][1] = value
                elif idx == 20 + 6 * ind:
                    calibration.chess_rotation_vector[ind][2] = value
                elif idx == 21 + 6 * ind:
                    calibration.chess_translation[ind][0] = value
                elif idx == 22 + 6 * ind:
                    calibration.chess_translation[ind][1] = value
                elif idx == 23 + 6 * ind:
                    calibration.chess_translation[ind][2] = value
        else:
            raise ValueError('Unknown i value: ' + str(idx))

    def getter(calibration, idx):
        if idx == 0:
            return [calibration.right_cam_rotation_vector[0]]
        elif idx == 1:
            return [calibration.right_cam_rotation_vector[1]]
        elif idx == 2:
            return [calibration.right_cam_rotation_vector[2]]
        elif idx == 3:
            return [calibration.right_cam_translation[0]]
        elif idx == 4:
            return [calibration.right_cam_translation[1]]
        elif idx == 5:
            return [calibration.right_cam_translation[2]]
        elif idx == 6:
            return [calibration.center_cam_rotation_vector[0]]
        elif idx == 7:
            return [calibration.center_cam_rotation_vector[1]]
        elif idx == 8:
            return [calibration.center_cam_rotation_vector[2]]
        elif idx == 9:
            return [calibration.center_cam_translation[0]]
        elif idx == 10:
            return [calibration.center_cam_translation[1]]
        elif idx == 11:
            return [calibration.center_cam_translation[2]]
        elif idx == 12:
            return [calibration.left_cam_rotation_vector[0]]
        elif idx == 13:
            return [calibration.left_cam_rotation_vector[1]]
        elif idx == 14:
            return [calibration.left_cam_rotation_vector[2]]
        elif idx == 15:
            return [calibration.left_cam_translation[0]]
        elif idx == 16:
            return [calibration.left_cam_translation[1]]
        elif idx == 17:
            return [calibration.left_cam_translation[2]]
        elif idx > 17:
            for ind1 in range(len(name_image_left)):
                if idx == 18 + 6 * ind1:
                    return [calibration.chess_rotation_vector[ind1][0]]
                elif idx == 19 + 6 * ind1:
                    return [calibration.chess_rotation_vector[ind1][1]]
                elif idx == 20 + 6 * ind1:
                    return [calibration.chess_rotation_vector[ind1][2]]
                elif idx == 21 + 6 * ind1:
                    return [calibration.chess_translation[ind1][0]]
                elif idx == 22 + 6 * ind1:
                    return [calibration.chess_translation[ind1][1]]
                elif idx == 23 + 6 * ind1:
                    return [calibration.chess_translation[ind1][2]]

        else:
            raise ValueError('Unknown i value: ' + str(idx))

    parameter_names = ['rr1', 'rr2', 'rr3', 'txr', 'tyr', 'tzr', 'rc1', 'rc2', 'rc3', 'txc', 'tyc', 'tzc', 'rl1', 'rl2', 'rl3', 'txl', 'tyl', 'tzl']
    for ind2 in range(len(name_image_left)):
        parameter_names.extend(['rchess1_' + str(ind2), 'rchess2_' + str(ind2), 'rchess3_' + str(ind2), 'txchess_' + str(ind2), 'tychess_' + str(ind2), 'tzchess_' + str(ind2)])

    for idx in range(0, 18 + 6 * len(name_image_left)):
        opt.pushParamScalar(group_name=parameter_names[idx], data_key='calibration', getter=partial(getter, idx=idx),
                            setter=partial(setter, idx=idx))

    # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(model):

        calibration = model['calibration']
        print("calibrations")
        print(calibration.left_cam_translation)
        error = []
        calibration.right_cam_image_points = []
        calibration.left_cam_image_points = []
        calibration.center_cam_image_points = []
        for ind in range(len(name_image_right)):
            right_cam_rotation_vector = np.zeros((1, 3), np.float32)
            right_cam_translation = np.zeros((3), np.float32)
            center_cam_rotation_vector = np.zeros((1, 3), np.float32)
            center_cam_translation = np.zeros((3), np.float32)
            left_cam_rotation_vector = np.zeros((1, 3), np.float32)
            left_cam_translation = np.zeros((3), np.float32)
            chess_rotation_vector = np.zeros((1, 3), np.float32)
            chess_translation = np.zeros((3), np.float32)

            for i in range(0, 3):
                right_cam_rotation_vector[0, i] = calibration.right_cam_rotation_vector[i][0]
                right_cam_translation[i] = calibration.right_cam_translation[i][0]
                center_cam_rotation_vector[0, i] = calibration.center_cam_rotation_vector[i][0]
                center_cam_translation[i] = calibration.center_cam_translation[i][0]
                left_cam_rotation_vector[0, i] = calibration.left_cam_rotation_vector[i][0]
                left_cam_translation[i] = calibration.left_cam_translation[i][0]
                chess_rotation_vector[0, i] = calibration.chess_rotation_vector[ind][i][0]
                chess_translation[i] = calibration.chess_translation[ind][i][0]

            # ---------------------------------------------------------
            # Get T from right cam to world (based on the params being optimized)
            # ---------------------------------------------------------
            right_cam_T_world = np.zeros((4, 4), np.float32)

            right_cam_T_world[0:3, 0:3], _ = cv2.Rodrigues(right_cam_rotation_vector)
            right_cam_T_world[0:3, 3] = right_cam_translation.T
            right_cam_T_world[3, :] = [0, 0, 0, 1]  # homogenize

            # ---------------------------------------------------------
            # Get T from center cam to world (based on the params being optimized)
            # ---------------------------------------------------------
            center_cam_T_world = np.zeros((4, 4), np.float32)

            center_cam_T_world[0:3, 0:3], _ = cv2.Rodrigues(center_cam_rotation_vector)
            center_cam_T_world[0:3, 3] = center_cam_translation.T
            center_cam_T_world[3, :] = [0, 0, 0, 1]  # homogenize

            # ---------------------------------------------------------
            # Get T from left cam to world (based on the params being optimized)
            # ---------------------------------------------------------
            left_cam_T_world = np.zeros((4, 4), np.float32)

            left_cam_T_world[0:3, 0:3], _ = cv2.Rodrigues(left_cam_rotation_vector)
            left_cam_T_world[0:3, 3] = left_cam_translation.T
            left_cam_T_world[3, :] = [0, 0, 0, 1]  # homogenize

            # ---------------------------------------------------------
            # Get T from chessboard to world
            # ---------------------------------------------------------
            chess_T_world = np.zeros((4, 4), np.float32)

            chess_T_world[0:3, 0:3], _ = cv2.Rodrigues(chess_rotation_vector)
            chess_T_world[0:3, 3] = chess_translation.T
            chess_T_world[3, :] = [0, 0, 0, 1]  # homogenize

            # ---------------------------------------------------------
            # Get T from world cam to chess (optimized)
            # ---------------------------------------------------------
            inv_chess_T_world = np.zeros((4, 4), np.float32)
            inv_chess_T_world[0:3, 0:3], inv_chess_T_world[0:3, 3:4] = inv_tranformation_matrix(
                chess_T_world[0:3, 0:3], chess_T_world[0:3, 3])
            inv_chess_T_world[3, :] = [0, 0, 0, 1]  # homogenize

            # ---------------------------------------------------------
            # Get aggregate T from right cam to chess (optimized)
            # ---------------------------------------------------------
            right_cam_T_chess_opt = np.matmul(right_cam_T_world, inv_chess_T_world)

            # ---------------------------------------------------------
            # Get aggregate T from center cam to chess (optimized)
            # ---------------------------------------------------------
            center_cam_T_chess_opt = np.matmul(center_cam_T_world, inv_chess_T_world)

            # ---------------------------------------------------------
            # Get aggregate T from center cam to chess (optimized)
            # ---------------------------------------------------------
            left_cam_T_chess_opt = np.matmul(left_cam_T_world, inv_chess_T_world)

            # ---------------------------------------------------------
            # Get T from right camera and center camera to chess (ground truth)
            # ---------------------------------------------------------
            rvec_cam_right, tvec_cam_right = find_cam_chess_realpoints(str(name_image_right[ind]), 1)
            right_cam_T_chess_ground_truth = np.zeros((3, 4), np.float32)
            right_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_right)[0]
            right_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_right.T

            rvec_cam_center, tvec_cam_center = find_cam_chess_realpoints(str(name_image_center[ind]), 2)
            center_cam_T_chess_ground_truth = np.zeros((3, 4), np.float32)
            center_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_center)[0]
            center_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_center.T

            rvec_cam_left, tvec_cam_left = find_cam_chess_realpoints(str(name_image_left[ind]), 0)
            left_cam_T_chess_ground_truth = np.zeros((3, 4), np.float32)
            left_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_left)[0]
            left_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_left.T

            # ---------------------------------------------------------
            # Draw projection of (optimized) 3D points
            # ---------------------------------------------------------
            tranformation_lists = []
            tranformation_lists.append(right_cam_T_chess_opt)
            tranformation_lists.append(center_cam_T_chess_opt)
            tranformation_lists.append(left_cam_T_chess_opt)

            for a3 in range(len(tranformation_lists)):
                transformation_matrix = tranformation_lists[a3]
                r_cam2tochess_vector, _ = cv2.Rodrigues(transformation_matrix[0:3, 0:3])
                # t_cam2tochess = np.zeros((3, 1))
                t_cam2tochess = transformation_matrix[0:3, 3]

                if a3 == 1:
                    imgpoints_optimize = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess,
                                                                 k_center, dist_center)

                    calibration.center_cam_image_points.append(imgpoints_optimize)

                    imgpoints_real = cv2.projectPoints(pts_chessboard, rvec_cam_center, tvec_cam_center, k_center,
                                                             dist_center)
                    if residuals_3d == 1:
                        for i in range(dimensions[0] * dimensions[1]):
                            points_op = np.matmul(np.array(transformation_matrix),
                                                  points_chessboard_homogeneous[:, i].reshape((4, 1)))
                            points_real_right = np.matmul(center_cam_T_chess_ground_truth,
                                                          points_chessboard_homogeneous[:, i].reshape((4, 1)))
                            error.append(math.sqrt(
                                (points_op[0] - points_real_right[0]) ** 2 + (
                                            points_op[1] - points_real_right[1]) ** 2 + (
                                        points_op[2] - points_real_right[2]) ** 2))
                    # print(calibration.center_cam_image_points[0][0][0])
                    # exit(0)

                elif a3 == 0:

                    imgpoints_optimize = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess, k_right,
                                                                 dist_right)

                    calibration.right_cam_image_points.append(imgpoints_optimize)

                    imgpoints_real = cv2.projectPoints(pts_chessboard, rvec_cam_right, tvec_cam_right, k_right,
                                                             dist_right)
                    if residuals_3d == 1:
                        for i in range(dimensions[0] * dimensions[1]):
                            points_op = np.matmul(np.array(transformation_matrix),
                                                  points_chessboard_homogeneous[:, i].reshape((4, 1)))
                            points_real_right = np.matmul(right_cam_T_chess_ground_truth,
                                                          points_chessboard_homogeneous[:, i].reshape((4, 1)))
                            error.append(math.sqrt(
                                (points_op[0] - points_real_right[0]) ** 2 + (
                                            points_op[1] - points_real_right[1]) ** 2 + (
                                        points_op[2] - points_real_right[2]) ** 2))
                elif a3 == 2:

                    imgpoints_optimize = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess, k_left,
                                                                 dist_left)

                    calibration.left_cam_image_points.append(imgpoints_optimize)

                    imgpoints_real = cv2.projectPoints(pts_chessboard, rvec_cam_left, tvec_cam_left, k_left,
                                                             dist_left)

                    if residuals_3d == 1:
                        for i in range(dimensions[0] * dimensions[1]):

                            points_op = np.matmul(transformation_matrix,
                                                  points_chessboard_homogeneous[:, i])
                            points_real_right = np.matmul(left_cam_T_chess_ground_truth,
                                                          points_chessboard_homogeneous[:, i])
                            dist_3d = math.sqrt(
                                (points_op[0] - points_real_right[0]) ** 2 + (
                                            points_op[1] - points_real_right[1]) ** 2 + (
                                        points_op[2] - points_real_right[2]) ** 2)
                            # print(dist_3d)
                            # exit(0)
                            error.append(dist_3d)

                    # print(imgpoints_optimize[0])
                    # print(imgpoints_real[0])
                    # exit(0)

                if residuals_3d == 0:
                    for a in range(dimensions[0] * dimensions[1]):

                        error.append(
                            math.sqrt((imgpoints_optimize[0][a][0][0] - imgpoints_real[0][a][0][0]) ** 2 + (
                                imgpoints_optimize[0][a][0][1] - imgpoints_real[0][a][0][1]) ** 2))


        # print("avg error: " + str(np.mean(np.array(error))))
        return error

    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------
    for a1 in range(len(name_image_left)):
        for a2 in range(number_sensors):
            for a in range(0, dimensions[0] * dimensions[1]):
                opt.pushResidual(name='r' + str(a) + '_s' + str(a2) + '_c' + str(a1), params=parameter_names)

    print('residuals = ' + str(opt.residuals))

    opt.computeSparseMatrix()
    opt.printSparseMatrix()

    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------

    # # fig = plt.figure()
    # # ax = fig.add_subplot(111)
    # fig = plt.figure()
    # ax = fig.gca()
    #
    # ax.set_xlabel('X'), ax.set_ylabel('Y'),
    # ax.set_xticklabels([]), ax.set_yticklabels([])
    # ax.set_xlim(-math.pi/2, math.pi/2), ax.set_ylim(-5, 5)
    #
    # # Draw cosine fucntion
    # f = np.cos(x)
    # ax.plot(x, f, label="cosine")
    # legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    #
    # y = 0 + \
    #     np.multiply(0, np.power(x, 1)) + \
    #     np.multiply(0, np.power(x, 2)) + \
    #     np.multiply(0, np.power(x, 3)) + \
    #     np.multiply(0, np.power(x, 4))
    #
    # handle_plot = ax.plot(x, y, label="calibration")
    # print(type(handle_plot))
    # print((handle_plot))
    #
    # wm = KeyPressManager.KeyPressManager.WindowManager(fig)
    # if wm.waitForKey(0., verbose=False):
    #     exit(0)

    # handles_out = {}
    # handles_out['point'] = ax.plot([pt_origin[0, 0], pt_origin[0, 0]], [pt_origin[1, 0], pt_origin[1, 0]],
    #                                [pt_origin[2, 0], pt_origin[2, 0]], 'k.')[0]
    # handles_out['text'] = ax.text(pt_origin[0, 0], pt_origin[1, 0], pt_origin[2, 0], text, color='black',
    #                               fontsize=fontsize)
    # else:
    #     handles['point'].set_xdata([pt_origin[0, 0], pt_origin[0, 0]])
    #     handles['point'].set_ydata([pt_origin[1, 0], pt_origin[1, 0]])
    #     handles['point'].set_3d_properties(zs=[pt_origin[2, 0], pt_origin[2, 0]])
    #
    #     handles['text'].set_position((pt_origin[0, 0], pt_origin[1, 0]))
    #     handles['text'].set_3d_properties(z=pt_origin[2, 0], zdir='x')

    def visualizationFunction(model):
        pass
        # for ind1 in range(len(name_image_right)):
        #     img1 = cv2.imread(name_image_right[ind1])
        #     img = cv2.drawChessboardCorners(img1, (9, 6), calibration.right_cam_image_points[ind1][0], True)
        #     cv2.imshow('img', img)
        #
        #     img2 = cv2.imread(name_image_center[ind1])
        #     img3 = cv2.drawChessboardCorners(img2, (9, 6), calibration.center_cam_image_points[ind1][0], True)
        #     cv2.imshow('img_center', img3)
        #
        #     img4 = cv2.imread(name_image_left[ind1])
        #     img5 = cv2.drawChessboardCorners(img4, (9, 6), calibration.left_cam_image_points[ind1][0], True)
        #     cv2.imshow('img_left', img5)
        #     cv2.waitKey(1)
        # wm = KeyPressManager.KeyPressManager.WindowManager()


    #     # y = calibration.param0[0] + \
    #     #     np.multiply(calibration.param1[0], np.power(x, 1)) + \
    #     #     np.multiply(calibration.param2[0], np.power(x, 2)) + \
    #     #     np.multiply(calibration.params_3_and_4[0][0], np.power(x, 3)) + \
    #     #     np.multiply(calibration.params_3_and_4[1][0], np.power(x, 4))
    #     #
    #     # handle_plot[0].set_ydata(y)
    # #
    #     wm = KeyPressManager.KeyPressManager.WindowManager(img)
    #     if wm.waitForKey(0.01, verbose=False):
    #         exit(0)
    # #
    opt.setVisualizationFunction(visualizationFunction, True)
    # calibration = model['calibration']

    # ---------------------------------------
    # --- Create X0 (First Guess)
    # ---------------------------------------
    # opt.fromXToData()
    # opt.callObjectiveFunction()
    # wm = KeyPressManager.KeyPressManager.WindowManager()
    # if wm.waitForKey():
    #     exit(0)
    #
    # ---------------------------------------
    # --- Start Optimization
    # ---------------------------------------
    print("\n\nStarting optimization")
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-8, 'diff_step': 1e-4})

    residuals_3d = 0 # Calcular erro final em pixeis
    error = []
    calibration.right_cam_image_points = []
    calibration.left_cam_image_points = []
    calibration.center_cam_image_points = []
    for ind in range(len(name_image_right)):
        right_cam_rotation_vector = np.zeros((1, 3), np.float32)
        right_cam_translation = np.zeros((3), np.float32)
        center_cam_rotation_vector = np.zeros((1, 3), np.float32)
        center_cam_translation = np.zeros((3), np.float32)
        left_cam_rotation_vector = np.zeros((1, 3), np.float32)
        left_cam_translation = np.zeros((3), np.float32)
        chess_rotation_vector = np.zeros((1, 3), np.float32)
        chess_translation = np.zeros((3), np.float32)

        for i in range(0, 3):
            right_cam_rotation_vector[0, i] = calibration.right_cam_rotation_vector[i][0]
            right_cam_translation[i] = calibration.right_cam_translation[i][0]
            center_cam_rotation_vector[0, i] = calibration.center_cam_rotation_vector[i][0]
            center_cam_translation[i] = calibration.center_cam_translation[i][0]
            left_cam_rotation_vector[0, i] = calibration.left_cam_rotation_vector[i][0]
            left_cam_translation[i] = calibration.left_cam_translation[i][0]
            chess_rotation_vector[0, i] = calibration.chess_rotation_vector[ind][i][0]
            chess_translation[i] = calibration.chess_translation[ind][i][0]

        # ---------------------------------------------------------
        # Get T from right cam to world (based on the params being optimized)
        # ---------------------------------------------------------
        right_cam_T_world = np.zeros((4, 4), np.float32)

        right_cam_T_world[0:3, 0:3], _ = cv2.Rodrigues(right_cam_rotation_vector)
        right_cam_T_world[0:3, 3] = right_cam_translation.T
        right_cam_T_world[3, :] = [0, 0, 0, 1]  # homogenize

        # ---------------------------------------------------------
        # Get T from center cam to world (based on the params being optimized)
        # ---------------------------------------------------------
        center_cam_T_world = np.zeros((4, 4), np.float32)

        center_cam_T_world[0:3, 0:3], _ = cv2.Rodrigues(center_cam_rotation_vector)
        center_cam_T_world[0:3, 3] = center_cam_translation.T
        center_cam_T_world[3, :] = [0, 0, 0, 1]  # homogenize

        # ---------------------------------------------------------
        # Get T from left cam to world (based on the params being optimized)
        # ---------------------------------------------------------
        left_cam_T_world = np.zeros((4, 4), np.float32)

        left_cam_T_world[0:3, 0:3], _ = cv2.Rodrigues(left_cam_rotation_vector)
        left_cam_T_world[0:3, 3] = left_cam_translation.T
        left_cam_T_world[3, :] = [0, 0, 0, 1]  # homogenize

        # ---------------------------------------------------------
        # Get T from chessboard to world
        # ---------------------------------------------------------
        chess_T_world = np.zeros((4, 4), np.float32)

        chess_T_world[0:3, 0:3], _ = cv2.Rodrigues(chess_rotation_vector)
        chess_T_world[0:3, 3] = chess_translation.T
        chess_T_world[3, :] = [0, 0, 0, 1]  # homogenize

        # ---------------------------------------------------------
        # Get T from world cam to chess (optimized)
        # ---------------------------------------------------------
        inv_chess_T_world = np.zeros((4, 4), np.float32)
        inv_chess_T_world[0:3, 0:3], inv_chess_T_world[0:3, 3:4] = inv_tranformation_matrix(
            chess_T_world[0:3, 0:3], chess_T_world[0:3, 3])
        inv_chess_T_world[3, :] = [0, 0, 0, 1]  # homogenize

        # ---------------------------------------------------------
        # Get aggregate T from right cam to chess (optimized)
        # ---------------------------------------------------------
        right_cam_T_chess_opt = np.matmul(right_cam_T_world, inv_chess_T_world)

        # ---------------------------------------------------------
        # Get aggregate T from center cam to chess (optimized)
        # ---------------------------------------------------------
        center_cam_T_chess_opt = np.matmul(center_cam_T_world, inv_chess_T_world)

        # ---------------------------------------------------------
        # Get aggregate T from center cam to chess (optimized)
        # ---------------------------------------------------------
        left_cam_T_chess_opt = np.matmul(left_cam_T_world, inv_chess_T_world)

        # ---------------------------------------------------------
        # Get T from right camera and center camera to chess (ground truth)
        # ---------------------------------------------------------
        rvec_cam_right, tvec_cam_right = find_cam_chess_realpoints(str(name_image_right[ind]), 1)
        right_cam_T_chess_ground_truth = np.zeros((3, 4), np.float32)
        right_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_right)[0]
        right_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_right.T

        rvec_cam_center, tvec_cam_center = find_cam_chess_realpoints(str(name_image_center[ind]), 2)
        center_cam_T_chess_ground_truth = np.zeros((3, 4), np.float32)
        center_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_center)[0]
        center_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_center.T

        rvec_cam_left, tvec_cam_left = find_cam_chess_realpoints(str(name_image_left[ind]), 0)
        left_cam_T_chess_ground_truth = np.zeros((3, 4), np.float32)
        left_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_left)[0]
        left_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_left.T

        # ---------------------------------------------------------
        # Draw projection of (optimized) 3D points
        # ---------------------------------------------------------
        tranformation_lists = []
        tranformation_lists.append(right_cam_T_chess_opt)
        tranformation_lists.append(center_cam_T_chess_opt)
        tranformation_lists.append(left_cam_T_chess_opt)

        for a3 in range(len(tranformation_lists)):
            transformation_matrix = tranformation_lists[a3]
            r_cam2tochess_vector, _ = cv2.Rodrigues(transformation_matrix[0:3, 0:3])
            # t_cam2tochess = np.zeros((3, 1))
            t_cam2tochess = transformation_matrix[0:3, 3]

            if a3 == 1:
                imgpoints_optimize = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess,
                                                       k_center, dist_center)

                calibration.center_cam_image_points.append(imgpoints_optimize)

                imgpoints_real = cv2.projectPoints(pts_chessboard, rvec_cam_center, tvec_cam_center, k_center,
                                                   dist_center)
                if residuals_3d == 1:
                    for i in range(dimensions[0] * dimensions[1]):
                        points_op = np.matmul(np.array(transformation_matrix),
                                              points_chessboard_homogeneous[:, i].reshape((4, 1)))
                        points_real_right = np.matmul(center_cam_T_chess_ground_truth,
                                                      points_chessboard_homogeneous[:, i].reshape((4, 1)))
                        error.append(math.sqrt(
                            (points_op[0] - points_real_right[0]) ** 2 + (
                                    points_op[1] - points_real_right[1]) ** 2 + (
                                    points_op[2] - points_real_right[2]) ** 2))
                # print(calibration.center_cam_image_points[0][0][0])
                # exit(0)

            elif a3 == 0:

                imgpoints_optimize = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess, k_right,
                                                       dist_right)

                calibration.right_cam_image_points.append(imgpoints_optimize)

                imgpoints_real = cv2.projectPoints(pts_chessboard, rvec_cam_right, tvec_cam_right, k_right,
                                                   dist_right)
                if residuals_3d == 1:
                    for i in range(dimensions[0] * dimensions[1]):
                        points_op = np.matmul(np.array(transformation_matrix),
                                              points_chessboard_homogeneous[:, i].reshape((4, 1)))
                        points_real_right = np.matmul(right_cam_T_chess_ground_truth,
                                                      points_chessboard_homogeneous[:, i].reshape((4, 1)))
                        error.append(math.sqrt(
                            (points_op[0] - points_real_right[0]) ** 2 + (
                                    points_op[1] - points_real_right[1]) ** 2 + (
                                    points_op[2] - points_real_right[2]) ** 2))
            elif a3 == 2:

                imgpoints_optimize = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess, k_left,
                                                       dist_left)

                calibration.left_cam_image_points.append(imgpoints_optimize)

                imgpoints_real = cv2.projectPoints(pts_chessboard, rvec_cam_left, tvec_cam_left, k_left,
                                                   dist_left)

                if residuals_3d == 1:
                    for i in range(dimensions[0] * dimensions[1]):
                        points_op = np.matmul(transformation_matrix,
                                              points_chessboard_homogeneous[:, i])
                        points_real_right = np.matmul(left_cam_T_chess_ground_truth,
                                                      points_chessboard_homogeneous[:, i])
                        dist_3d = math.sqrt(
                            (points_op[0] - points_real_right[0]) ** 2 + (
                                    points_op[1] - points_real_right[1]) ** 2 + (
                                    points_op[2] - points_real_right[2]) ** 2)
                        # print(dist_3d)
                        # exit(0)
                        error.append(dist_3d)

                # print(imgpoints_optimize[0])
                # print(imgpoints_real[0])
                # exit(0)

            if residuals_3d == 0:
                for a in range(dimensions[0] * dimensions[1]):
                    error.append(
                        math.sqrt((imgpoints_optimize[0][a][0][0] - imgpoints_real[0][a][0][0]) ** 2 + (
                                imgpoints_optimize[0][a][0][1] - imgpoints_real[0][a][0][1]) ** 2))
    print('Erro medio final em pixeis:')
    print(np.mean(error))
    print('############')
if wm.waitForKey():
        exit(0)
