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
        self.cams_rotation_vector = []
        self.cams_translation = []


def generate_chessboard(size, dimensions):
    objp = np.zeros((dimensions[0] * dimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2)
    objp = objp * size
    return objp


def create_matrix_4by4(rvec, tvec):
    matrix = np.zeros((4, 4), np.float32)
    matrix[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    matrix[0:3, 3] = tvec.T
    matrix[3, :] = [0, 0, 0, 1]  # homogenize
    return matrix


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
        retval, rvec, tvec = cv2.solvePnP(objpoints[0], imgpoints[0], k_matrix, dist_matrix)
    else:
        return 0

    return rvec, tvec


def inv_tranformation_matrix_vectors(r, t):
    x_array = np.array(r[0])
    r_inv = r.T
    t_inv = np.zeros((3, 1), np.float32)
    t_inv[0] = -np.dot(r[0:3, 0], t)
    t_inv[1] = -np.dot(r[0:3, 1], t)
    t_inv[2] = -np.dot(r[0:3, 2], t)

    return r_inv, t_inv


def inv_transformation_matrix(matrix):
    r = matrix[0:3, 0:3]
    t = matrix[0:3, 3]
    r_inv = matrix[0:3, 0:3].T
    t_inv = np.zeros((3, 1), np.float32)
    t_inv[0] = -np.dot(r[0:3, 0], t)
    t_inv[1] = -np.dot(r[0:3, 1], t)
    t_inv[2] = -np.dot(r[0:3, 2], t)
    inv_matrix = np.zeros((4, 4))
    inv_matrix[0:3, 0:3] = r_inv
    inv_matrix[0:3, 3:4] = t_inv
    inv_matrix[3, :] = [0, 0, 0, 1]  # homogenize

    return inv_matrix

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

    calibration = Calibration()
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
    i_collections = 0
    i_initialize = 0
    i_vetor = np.zeros(len(sensors))
    sensor = []
    cams_first_guess = np.zeros((len(sensors), 4, 4), np.float32)
    for i_1 in range(len(sensors)):
        calibration.cams_rotation_vector.append([0.0, 0.0, 0.0])
        calibration.cams_translation.append([0.0, 0.0, 0.0])
    # for i in calibration_data['collections']:
    for i in range(len(calibration_data['collections'])):
        i2 = 0
        check_chess = 0
        for i1 in sensors:
            name_image = args['dataset_path'] + calibration_data['collections'][str(i)]['data'][str(i1)]['data_file']
            name_image_list.append(name_image)

            if i_initialize == 0:
                sensor.append(str(i1))
                k_matrix[i2, 0, :] = calibration_data['sensors'][str(i1)]['camera_info']['K'][0:3]
                k_matrix[i2, 1, :] = calibration_data['sensors'][str(i1)]['camera_info']['K'][3:6]
                k_matrix[i2, 2, :] = calibration_data['sensors'][str(i1)]['camera_info']['K'][6:9]
                dist_matrix[i2, :] = calibration_data['sensors'][str(i1)]['camera_info']['D']
                dist_matrix[1, :] = [0.0, 0.0, 0.0, 0.0, 0.0]

            if i_vetor[i2] == 0:
                # -------------------------------------------
                # ----- First Guess
                # -------------------------------------------
                if calibration_data['collections'][str(i)]['data'][str(i1)]['detected'] == 1:
                    rvec_cam, tvec_cam = find_cam_chess_realpoints(str(name_image), k_matrix[i2, :, :], dist_matrix[i2, :])
                    cams_first_guess[i2, 0:4, 0:4] = create_matrix_4by4(rvec_cam, tvec_cam)


                    i_vetor[i2] = 1

            if check_chess == 0 and calibration_data['collections'][str(i)]['data'][str(i1)]['detected'] == 1:

                i_chess = i2 * 1
                name_image_chess = str(name_image)
                check_chess = 1


            i2 += 1

        rvec_cam_chess, tvec_cam_chess = find_cam_chess_realpoints(name_image_chess, k_matrix[i_chess, :, :], dist_matrix[i_chess, :])
        chess_ground_truth = create_matrix_4by4(rvec_cam_chess, tvec_cam_chess)
        if i_chess > 0:
            second_cam_matrix = inv_transformation_matrix(cams_first_guess[i_chess, :, :])
            # print(cams_first_guess[i_chess, :, :])
            # print(inv_transformation_matrix(second_cam_matrix))
            # exit(0)
            chess_ground_truth = np.dot(np.dot(cams_first_guess[0, :, :], second_cam_matrix), chess_ground_truth)
        chess_first_guess_rotation, chess_first_guess_translation = inv_tranformation_matrix_vectors(
            chess_ground_truth[0:3, 0:3],
            chess_ground_truth[0:3, 3])
        if i == 0:
            first_chess_first_guess = inv_transformation_matrix(chess_ground_truth)
        calibration.chess_rotation_vector.append([0.0, 0.0, 0.0])
        calibration.chess_translation.append([0.0, 0.0, 0.0])
        calibration.chess_rotation_vector[i_collections][0] = cv2.Rodrigues(chess_first_guess_rotation)[0][0]
        calibration.chess_rotation_vector[i_collections][1] = cv2.Rodrigues(chess_first_guess_rotation)[0][1]
        calibration.chess_rotation_vector[i_collections][2] = cv2.Rodrigues(chess_first_guess_rotation)[0][2]

        calibration.chess_translation[i_collections][0] = chess_first_guess_translation.T[0][0]
        calibration.chess_translation[i_collections][1] = chess_first_guess_translation.T[0][1]
        calibration.chess_translation[i_collections][2] = chess_first_guess_translation.T[0][2]
        i_collections += 1
        i_initialize += 1

    for ind6 in range(len(sensors)):
        cams_first_guess[ind6, :, :] = np.dot(cams_first_guess[ind6, :, :], first_chess_first_guess)
        calibration.cams_rotation_vector[ind6][0] = cv2.Rodrigues(cams_first_guess[ind6, 0:3, 0:3])[0][0]
        calibration.cams_rotation_vector[ind6][1] = cv2.Rodrigues(cams_first_guess[ind6, 0:3, 0:3])[0][1]
        calibration.cams_rotation_vector[ind6][2] = cv2.Rodrigues(cams_first_guess[ind6, 0:3, 0:3])[0][2]

        calibration.cams_translation[ind6][0] = cams_first_guess[ind6, 0:3, 3:4].T[0][0]
        calibration.cams_translation[ind6][1] = cams_first_guess[ind6, 0:3, 3:4].T[0][1]
        calibration.cams_translation[ind6][2] = cams_first_guess[ind6, 0:3, 3:4].T[0][2]

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('calibration', calibration)


 # Create specialized getter and setter functions
    def setter(calibration, value, idx):
        if idx < 6 * len(sensors):
            for ind in range(len(sensors)):
                if idx == 0 + 6 * ind:
                    calibration.cams_rotation_vector[ind][0] = value
                elif idx == 1 + 6 * ind:
                    calibration.cams_rotation_vector[ind][1] = value
                elif idx == 2 + 6 * ind:
                    calibration.cams_rotation_vector[ind][2] = value
                elif idx == 3 + 6 * ind:
                    calibration.cams_translation[ind][0] = value
                elif idx == 4 + 6 * ind:
                    calibration.cams_translation[ind][1] = value
                elif idx == 5 + 6 * ind:
                    calibration.cams_translation[ind][2] = value
        elif idx >= 6 * len(sensors):
            for ind in range(len(name_image_list)/len(sensors)):
                if idx == 6 * len(sensors) + 6 * ind:
                    calibration.chess_rotation_vector[ind][0] = value
                elif idx == 6 * len(sensors) + 1 + 6 * ind:
                    calibration.chess_rotation_vector[ind][1] = value
                elif idx == 6 * len(sensors) + 2 + 6 * ind:
                    calibration.chess_rotation_vector[ind][2] = value
                elif idx == 6 * len(sensors) + 3 + 6 * ind:
                    calibration.chess_translation[ind][0] = value
                elif idx == 6 * len(sensors) + 4 + 6 * ind:
                    calibration.chess_translation[ind][1] = value
                elif idx == 6 * len(sensors) + 5 + 6 * ind:
                    calibration.chess_translation[ind][2] = value
        else:
            raise ValueError('Unknown i value: ' + str(idx))

    def getter(calibration, idx):
        if idx < 6 * len(sensors):
            for ind1 in range(len(sensors)):
                if idx == 0 + 6 * ind1:
                    return [calibration.cams_rotation_vector[ind1][0]]
                elif idx == 1 + 6 * ind1:
                    return [calibration.cams_rotation_vector[ind1][1]]
                elif idx == 2 + 6 * ind1:
                    return [calibration.cams_rotation_vector[ind1][2]]
                elif idx == 3 + 6 * ind1:
                    return [calibration.cams_translation[ind1][0]]
                elif idx == 4 + 6 * ind1:
                    return [calibration.cams_translation[ind1][1]]
                elif idx == 5 + 6 * ind1:
                    return [calibration.cams_translation[ind1][2]]
            return [calibration.right_cam_rotation_vector[0]]
        elif idx >= 6 * len(sensors):
            for ind1 in range(len(name_image_list) / len(sensors)):
                if idx == 6 * len(sensors) + 6 * ind1:
                    return [calibration.chess_rotation_vector[ind1][0]]
                elif idx == 6 * len(sensors) + 1 + 6 * ind1:
                    return [calibration.chess_rotation_vector[ind1][1]]
                elif idx == 6 * len(sensors) + 2 + 6 * ind1:
                    return [calibration.chess_rotation_vector[ind1][2]]
                elif idx == 6 * len(sensors) + 3 + 6 * ind1:
                    return [calibration.chess_translation[ind1][0]]
                elif idx == 6 * len(sensors) + 4 + 6 * ind1:
                    return [calibration.chess_translation[ind1][1]]
                elif idx == 6 * len(sensors) + 5 + 6 * ind1:
                    return [calibration.chess_translation[ind1][2]]

        else:
            raise ValueError('Unknown i value: ' + str(idx))
    parameter_names = []
    for ind2 in range(len(sensors)):
        parameter_names.extend(['r1' + str(ind2), 'r2' + str(ind2), 'r3' + str(ind2), 't1' + str(ind2), 't2' + str(ind2), 't3' + str(ind2)])

    for ind2 in range(len(name_image_list)/len(sensors)):
        parameter_names.extend(['rchess1_' + str(ind2), 'rchess2_' + str(ind2), 'rchess3_' + str(ind2), 'txchess_' + str(ind2), 'tychess_' + str(ind2), 'tzchess_' + str(ind2)])

    for idx in range(0, 6 * (len(name_image_list) / len(sensors) + len(sensors))):
        opt.pushParamScalar(group_name=parameter_names[idx], data_key='calibration', getter=partial(getter, idx=idx),
                            setter=partial(setter, idx=idx))

   # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(model):

        calibration = model['calibration']
        error = []
        calibration.right_cam_image_points = []
        calibration.left_cam_image_points = []
        calibration.center_cam_image_points = []
        cams_rotation_vector = np.zeros((len(sensors), 1, 3), np.float32)
        cams_translation = np.zeros((len(sensors), 3), np.float32)
        chess_rotation_vector = np.zeros((1, 3), np.float32)
        chess_translation = np.zeros((3), np.float32)
        for ind in range(len(name_image_list)/len(sensors)):

            for ind5 in range(0, 3):
                for i2 in range(len(sensors)):
                    # print(calibration.cams_rotation_vector[i2])
                    # exit(0)
                    cams_rotation_vector[i2, 0, ind5] = calibration.cams_rotation_vector[i2][ind5][0]
                    cams_translation[i2, ind5] = calibration.cams_translation[i2][ind5][0]
                chess_rotation_vector[0, ind5] = calibration.chess_rotation_vector[ind][ind5][0]
                chess_translation[ind5] = calibration.chess_translation[ind][ind5][0]

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
            inv_chess_T_world[0:3, 0:3], inv_chess_T_world[0:3, 3:4] = inv_tranformation_matrix_vectors(
                chess_T_world[0:3, 0:3], chess_T_world[0:3, 3])
            inv_chess_T_world[3, :] = [0, 0, 0, 1]  # homogenize

            # ---------------------------------------------------------
            # Get T from right cam to world (based on the params being optimized)
            # ---------------------------------------------------------
            cams_T_world = np.zeros((len(sensors), 4, 4), np.float32)
            ind3 = 0
            for i in sensors:
                if calibration_data['collections'][str(ind)]['data'][str(i)]['detected'] == 1:
                    cams_T_world = create_matrix_4by4(cams_rotation_vector[ind3, :, :], cams_translation[ind3, :])
                    transformation_matrix = np.dot(cams_T_world, inv_chess_T_world)
                    rvec_cam_right, tvec_cam_right = find_cam_chess_realpoints(str(name_image_list[ind * 3 + ind3]), k_matrix[ind3, :, :], dist_matrix[ind3, :])
                    right_cam_T_chess_ground_truth = np.zeros((3, 4), np.float32)
                    right_cam_T_chess_ground_truth[0:3, 0:3] = cv2.Rodrigues(rvec_cam_right)[0]
                    right_cam_T_chess_ground_truth[0:3, 3] = tvec_cam_right.T
                    r_cam2tochess_vector, _ = cv2.Rodrigues(transformation_matrix[0:3, 0:3])
                    t_cam2tochess = transformation_matrix[0:3, 3]
                    imgpoints_optimize = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess,
                                                                 k_matrix[ind3, :, :], dist_matrix[ind3, :])

                    calibration.center_cam_image_points.append(imgpoints_optimize)

                    imgpoints_real = cv2.projectPoints(pts_chessboard, rvec_cam_right, tvec_cam_right, k_matrix[ind3, :, :],
                                                             dist_matrix[ind3, :])
                    for a in range(dimensions[0] * dimensions[1]):

                        error.append(
                            math.sqrt((imgpoints_optimize[0][a][0][0] - imgpoints_real[0][a][0][0]) ** 2 + (
                                imgpoints_optimize[0][a][0][1] - imgpoints_real[0][a][0][1]) ** 2))



                ind3 += 1

        # print("avg error: " + str(np.mean(np.array(error))))
        return error

    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------

    for a1 in range(len(name_image_list)/len(sensors)):
        ind4 = 0
        for a2 in sensors:
            if calibration_data['collections'][str(a1)]['data'][str(a2)]['detected'] == 1:
                for a in range(0, dimensions[0] * dimensions[1]):
                    opt.pushResidual(name='r' + str(a) + '_s' + str(ind4) + '_c' + str(a1), params=parameter_names)
            ind4 += 1

    print('residuals = ' + str(opt.residuals))

    opt.computeSparseMatrix()
    opt.printSparseMatrix()

    # ---------------------------------------
    # --- Define THE VISUALIZATION FUNCTION
    # ---------------------------------------

    def visualizationFunction(model):
        pass


    opt.setVisualizationFunction(visualizationFunction, True)

    print("\n\nStarting optimization")
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'diff_step': 1e-8})



