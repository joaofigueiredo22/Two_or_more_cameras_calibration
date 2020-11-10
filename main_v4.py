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
from collections import namedtuple

from colorama import Fore


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

class Calibration:

    def __init__(self, sensors, collections, sensor_info):
        self.right_cam_rotation_vector = [0.0, 0.0, 0.0]
        self.center_cam_rotation_vector = [0.0, 0.0, 0.0]
        self.left_cam_rotation_vector = [0.0, 0.0, 0.0]
        self.right_cam_translation = [0.0, 0.0, 0.0]
        self.center_cam_translation = [0.0, 0.0, 0.0]
        self.left_cam_translation = [0.0, 0.0, 0.0]
        self.chess_rotation_vector = np.zeros((len(collections), 3))
        self.chess_translation = np.zeros((len(collections), 3))
        self.right_cam_image_points = []
        self.center_cam_image_points = []
        self.left_cam_image_points = []
        self.cams_rotation_vector = np.zeros((len(sensors), 3))
        self.cams_translation = np.zeros((len(sensors), 3))
        self.sensors = sensor_info
        self.collections = collections

    def __getitem__(self, key):
        print ("Inside `__getitem__` method!")
        return self

    def showResults(self, sensors_first_guess):
        for i, s in enumerate(self.sensors):
            for ind, col in enumerate(self.collections):
                # Image First Guess
                chess_matrix = chess_first_guess[ind, :, :]
                world_to_chess = inv_transformation_matrix(chess_matrix)
                transformation_matrix = np.dot(sensors_first_guess[i, :, :], world_to_chess)
                r_cam2tochess_vector, t_cam2tochess = matrix_to_vector(transformation_matrix)

                image_points_first_guess = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess,
                                                       s.k, s.dist)

                img1 = cv2.imread(s.image + str(col) + '.jpg')
                img2 = cv2.imread(s.image + str(col) + '.jpg')

                img_first_guess = cv2.drawChessboardCorners(img1, (9, 6), image_points_first_guess[0], True)

                chess_otimized = create_matrix_4by4(self.chess_rotation_vector[ind, :], self.chess_translation[ind, :])
                world_to_chess_otimized = inv_transformation_matrix(chess_otimized)
                cam_matrix_otimized = create_matrix_4by4(self.cams_rotation_vector[i, :], self.cams_translation[i, :])
                transformation_matrix_otimized = np.dot(cam_matrix_otimized, world_to_chess_otimized)
                rotation_vector_otimized, translation_vector_otimized = matrix_to_vector(transformation_matrix_otimized)
                image_points_otimized = cv2.projectPoints(pts_chessboard, rotation_vector_otimized, translation_vector_otimized, s.k, s.dist)
                img_otimized = cv2.drawChessboardCorners(img2, (9, 6), image_points_otimized[0], True)
                img = cv2.hconcat((img_first_guess, img_otimized))

                cv2.imshow('Sensor: ' + s.name + ' Collection: ' + str(col), img)
                cv2.waitKey(3000)
                cv2.destroyWindow('Sensor: ' + s.name + ' Collection: ' + str(col))

    def resetImagePoints(self):
        for i, s in enumerate(self.sensors):
            self.sensors[i].image_points = []
            self.sensors[i].image_index = 0


class CollectionSensorPair:

    def __init__(self, collection, s1, s2):
        pair = namedtuple('pair', ['s1', 's2'])

        self.collection = collection
        self.pair = pair(s1, s2)

    def __str__(self):
        return "(collection: " + str(self.collection) + ", sensor pair: " + self.pair.s1 + " - " + self.pair.s2 + ")"


class Sensor:
    def __init__(self, k, dist, name, name_image):
        self.k = k
        self.dist = dist
        self.index_first_guess = 0
        self.name = name
        self.image = name_image
        self.image_points = []
        self.rotation_vector = [0.0, 0.0, 0.0]
        self.translation_vector = [0.0, 0.0, 0.0]
        self.image_index = 0

    def __str__(self):
        return "Sensor: " + str(self.name) + " \n \nhas k matrix: \n" + str(self.k) + " \n \n" + "and d matrix: \n" + str(self.dist) + ""


class FirtsGuessCollections:

    def __init__(self, dict):
        self.dict = dict
        self.matrix = np.zeros((len(self.dict['collections']), len(self.dict['calibration_config']['sensor_order'])))
        self.collections_sensors_list = []
        self.valid_collections_list = []
        self.non_paired_sensors = []
        self.sensors_list = self.dict['calibration_config']['sensor_order']
        self.sensors = []
        self.world_referencial_connection_list = []
        self.sensors_first_guess = np.zeros((len(self.sensors_list), 4, 4))

    def Sensorsclass(self, args):

        for i, s in enumerate(self.sensors_list):
            a = np.zeros(len(self.dict['sensors'][str(s)]['camera_info']['K']))
            for i_3 in range(len(self.dict['sensors'][str(s)]['camera_info']['K'])):
                a[i_3] = float(self.dict['sensors'][str(s)]['camera_info']['K'][str(i_3)])
            k = a.reshape(3, 3)
            dist = np.zeros((5), np.float64)
            for i_3 in range(len(self.dict['sensors'][str(s)]['camera_info']['D'])):
                dist[i_3] = float(self.dict['sensors'][str(s)]['camera_info']['D'][str(i_3)])

            name_img = str(args['dataset_path'] + s + '_')
            new = Sensor(k, dist, s, name_img)
            self.sensors.append(new)
        return self.sensors

    def detectedChessCollectionsPerSensor(self):
        for i in range(len(self.dict['collections'])):
            i2 = 0
            for i1 in self.dict['calibration_config']['sensor_order']:
                self.matrix[i, i2] = self.dict['collections'][str(i)]['data'][str(i1)]['detected']
                i2 += 1
            if sum(self.matrix[i, :]) > 1:
                self.valid_collections_list.append(i)

    def filterDetectedCollactionsMatrix(self):
        self.filtered_matrix = np.zeros((len(self.valid_collections_list), self.matrix.shape[1]))
        for i in range(len(self.valid_collections_list)):
            self.filtered_matrix[i, :] = self.matrix[self.valid_collections_list[i], :]

    def addCollectionSensorPair(self, collection, s1, s2):
        new = CollectionSensorPair(collection, s1, s2)
        self.collections_sensors_list.append(new)

    def createCollectionPairSensorList_v2(self):

        self.sensors_list = self.dict['calibration_config']['sensor_order']

        self.world_referencial_connection_list.append(self.sensors_list[0])
        x = 0
        while x == 0:
            size_list = len(self.world_referencial_connection_list)
            for index, s in enumerate(self.sensors_list[1:]):
                if s in self.world_referencial_connection_list:
                    break
                for index_1, col in enumerate(self.valid_collections_list):
                    if self.filtered_matrix[index_1, index + 1] == 1:
                        for i, s1 in enumerate(self.world_referencial_connection_list):
                            if self.filtered_matrix[index_1, self.sensors_list.index(s1)] == 1:
                                self.addCollectionSensorPair(col, s1, s)
                                self.world_referencial_connection_list.append(s)
                                if len(self.world_referencial_connection_list) == len(self.sensors_list):
                                    x = 1
                                    return
                                break
                        if s in self.world_referencial_connection_list:
                            break

            if size_list == len(self.world_referencial_connection_list):
                for s in self.sensors_list:
                    sensor_recognized = False

                    for index, value in enumerate(self.collections_sensors_list):
                        if value.pair.s1 == s or value.pair.s2 == s:
                            sensor_recognized = True

                    if not sensor_recognized:
                        print("\n")
                        print(
                                Fore.RED + "ERROR GRAVISSIMO: " + Fore.RESET + "There is, at least, one sensor tandem that doesn't recognize the same chessboard collection.")
                        print("Please, provide the requested information to compute the first guess.")
                        print("\n")
                        exit(0)
                print(
                            Fore.GREEN + "\nI've managed to fix the issue, all set to initialize the first guess!" + Fore.RESET)
                x = 1
                return

    def createTransformationmatrix(self, args):

        self.sensors_first_guess[0, :, :] = np.eye(4)
        for i, s in enumerate(self.collections_sensors_list):
            s1_image_name = str(args['dataset_path'] + s.pair.s1 + '_' + str(s.collection) + '.jpg')
            s2_image_name = str(args['dataset_path'] + s.pair.s2 + '_' + str(s.collection) + '.jpg')

            s1_k = self.sensors[self.sensors_list.index(s.pair.s1)].k
            s1_dist = self.sensors[self.sensors_list.index(s.pair.s1)].dist
            s2_k = self.sensors[self.sensors_list.index(s.pair.s2)].k
            s2_dist = self.sensors[self.sensors_list.index(s.pair.s2)].dist

            self.sensors[self.sensors_list.index(s.pair.s2)].index_first_guess = s.collection
            if i == 0:
                self.sensors[self.sensors_list.index(s.pair.s1)].index_first_guess = s.collection

            s1_rvec, s1_tvec = get_solvepnp(s1_image_name, s1_k, s1_dist)
            s2_rvec, s2_tvec = get_solvepnp(s2_image_name, s2_k, s2_dist)

            s1_transformation_matrix = create_matrix_4by4(s1_rvec, s1_tvec)
            s2_transformation_matrix = create_matrix_4by4(s2_rvec, s2_tvec)

            s1_inv_matrix = inv_transformation_matrix(s1_transformation_matrix)

            if self.sensors_list.index(s.pair.s1) == 0:
                self.sensors_first_guess[self.sensors_list.index(s.pair.s2), :, :] = np.dot(s2_transformation_matrix, s1_inv_matrix)
            else:
                aux_transformation_matrix = np.dot(s2_transformation_matrix, s1_inv_matrix)
                self.sensors_first_guess[self.sensors_list.index(s.pair.s2), :, :] = np.dot(aux_transformation_matrix, self.sensors_first_guess[self.sensors_list.index(s.pair.s1), :, :])
        return self.sensors_first_guess

    def chessboardsFirstGuess(self):
        chess_first_guess = np.zeros((len(self.valid_collections_list), 4, 4))
        for i, col in enumerate(self.valid_collections_list):
            # print(self.filtered_matrix[0].index(1))
            index_sensor = np.where(self.filtered_matrix[i, :] == 1)[0][0]
            image = str(self.sensors[index_sensor].image + str(col) + '.jpg')
            k = self.sensors[index_sensor].k
            dist = self.sensors[index_sensor].dist
            chess_rvec, chess_tvec = get_solvepnp(image, k, dist)
            cam_to_chess = create_matrix_4by4(chess_rvec, chess_tvec)
            chess_to_cam = inv_transformation_matrix(cam_to_chess)
            chess_to_world = np.dot(chess_to_cam, self.sensors_first_guess[index_sensor, :, :])
            chess_first_guess[i, :, :] = chess_to_world

        return chess_first_guess

    def createCollectionPairSensorList(self):

        for i1 in self.dict['calibration_config']['sensor_order']:
        # for i1 in self.dict['sensors']:
            self.sensors_list.append(i1)

        for i in range(len(self.sensors_list) - 1):
            for index, col in enumerate(self.valid_collections_list):

                if sum(self.filtered_matrix[index, i:i+2]) == 2:
                    self.addCollectionSensorPair(col, self.sensors_list[i], self.sensors_list[i+1])
                    break

        if len(self.collections_sensors_list) < len(self.sensors_list) - 1:
            print(Fore.RED + "\nThere is a issue.... Im going to work on it now." + Fore.RESET)

            for s in self.sensors_list:
                sensor_recognized = False

                for index, value in enumerate(self.collections_sensors_list):
                    if value.pair.s1 == s or value.pair.s2 == s:
                        sensor_recognized = True

                if not sensor_recognized:
                    self.non_paired_sensors.append(s)
                    non_paired_sensors_index = self.sensors_list.index(s)
                    for index, col in enumerate(self.valid_collections_list):
                        if self.filtered_matrix[index, non_paired_sensors_index] == 1:
                            good_collection = True
                            all_sensors_recognizing_chess_index = [i for i, x in enumerate(self.filtered_matrix[index, :]) if x == 1]
                            # print(self.filtered_matrix[:])
                            # print(all_sensors_recognizing_chess_index)
                            sensors_recognizing_chess_index = [x for i, x in enumerate(all_sensors_recognizing_chess_index) if x != non_paired_sensors_index]
                            if sensors_recognizing_chess_index[0] < non_paired_sensors_index:
                                self.addCollectionSensorPair(col, self.sensors_list[sensors_recognizing_chess_index[0]], self.sensors_list[non_paired_sensors_index])
                            else:
                                self.addCollectionSensorPair(col, self.sensors_list[non_paired_sensors_index], self.sensors_list[sensors_recognizing_chess_index[0]])
                            break

            for s in self.sensors_list:
                sensor_recognized = False

                for index, value in enumerate(self.collections_sensors_list):
                    if value.pair.s1 == s or value.pair.s2 == s:
                        sensor_recognized = True

                if not sensor_recognized:
                    print("\n")
                    print(
                                Fore.RED + "ERROR GRAVISSIMO: " + Fore.RESET + "There is, at least, one sensor tandem that doesn't recognize the same chessboard collection.")
                    print("Please, provide the requested information to compute the first guess.")
                    print("\n")
                    exit(0)
            print(Fore.GREEN + "\nI've managed to fix the issue, all set to initialize the first guess!" + Fore.RESET)

    def showFirstGuessCorners(self, chess_first_guess):

        for i, s in enumerate(self.sensors):
            col = s.index_first_guess
            chess_matrix = chess_first_guess[self.valid_collections_list.index(col), :, :]
            world_to_chess = inv_transformation_matrix(chess_matrix)
            transformation_matrix = np.dot(self.sensors_first_guess[i, :, :], world_to_chess)
            r_cam2tochess_vector, t_cam2tochess = matrix_to_vector(transformation_matrix)

            image_points = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess,
                                                   s.k, s.dist)

            img1 = cv2.imread(s.image + str(col) + '.jpg')

            img1 = cv2.drawChessboardCorners(img1, (9, 6), image_points[0], True)
            cv2.imshow('img_' + s.name, img1)

            cv2.waitKey(1000)

    def __str__(self):

        string="["
        for i in self.collections_sensors_list:
            string = string + str(i.collection) + ", (" + str(i.pair.s1) + ", " + str(i.pair.s2) + ")" + "\n" if i != self.collections_sensors_list[-1] else string + str(i.collection) + ", (" + str(i.pair.s1) + ", " + str(i.pair.s2) + ")"
        string = string + "]"

        return string


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


def matrix_to_vector(matrix):
    rvec = cv2.Rodrigues(matrix[0:3, 0:3])[0].T
    tvec = matrix[0:3, 3].T
    return rvec, tvec


def find_chess_points(fname):
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
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
    return imgpoints


def get_solvepnp(fname, k_matrix, dist_matrix):
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
    t_inv[0] = -np.dot(t, r[0:3, 0])
    t_inv[1] = -np.dot(t, r[0:3, 1])
    t_inv[2] = -np.dot(t, r[0:3, 2])
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

    json_file = args['json_path']

    # Image used
    if not args['dataset_path'][-1] == '/':  # make sure the path is correct
        args['dataset_path'] += '/'

    f = open(args['json_path'], 'r')
    calibration_data = json.load(f)
    # dimensions = [0, 0]
    dimensions = calibration_data['calibration_config']['calibration_pattern']['dimension']
    # for i in range(len(calibration_data['calibration_config']['calibration_pattern']['dimension'])):
    #     dimensions[i] = calibration_data['calibration_config']['calibration_pattern']['dimension'][str(i)]
    size_board = calibration_data['calibration_config']['calibration_pattern']['size']

    pts_chessboard = generate_chessboard(size_board, dimensions)
    sensors = calibration_data['calibration_config']['sensor_order']

    # -------------------------------------
    # First cam First guess
    # -------------------------------------
    # See new Class

    classe_importante = FirtsGuessCollections(calibration_data)

    info_sensors = classe_importante.Sensorsclass(args)
    classe_importante.detectedChessCollectionsPerSensor()

    # print("\nclasse_importante.matrix")
    # print(classe_importante.matrix)

    classe_importante.filterDetectedCollactionsMatrix()

    # print("\nclasse_importante.valid_collections_list")
    # print(classe_importante.valid_collections_list)
    #
    # print("\nclasse_importante.filtered_matrix")
    # print(classe_importante.filtered_matrix)

    classe_importante.createCollectionPairSensorList_v2()

    # print("\nclasse_importante.sensors_list")
    # print(classe_importante.sensors_list)
    #
    # print("\nclasse_importante.non_paired_sensors")
    # print(classe_importante.non_paired_sensors)
    #
    # print ("\nclasse_importante")
    # print (classe_importante)

    cams_first_guess = classe_importante.createTransformationmatrix(args)

    chess_first_guess = classe_importante.chessboardsFirstGuess()

    classe_importante.showFirstGuessCorners(chess_first_guess)

    print ("\nChessboards First Guess")
    print (cams_first_guess)

    calibration = Calibration(classe_importante.sensors_list, classe_importante.valid_collections_list, info_sensors)

    list_detected = classe_importante.valid_collections_list

    for ind6 in range(len(sensors)):
        rotation_vector, translation_vector = matrix_to_vector(cams_first_guess[ind6, :, :])

        calibration.sensors[ind6].rotation_vector[0] = rotation_vector[0][0]
        calibration.sensors[ind6].rotation_vector[1] = rotation_vector[0][1]
        calibration.sensors[ind6].rotation_vector[2] = rotation_vector[0][2]

        calibration.sensors[ind6].translation_vector[0] = translation_vector[0]
        calibration.sensors[ind6].translation_vector[1] = translation_vector[1]
        calibration.sensors[ind6].translation_vector[2] = translation_vector[2]

        calibration.cams_rotation_vector[ind6][0] = rotation_vector[0][0]
        calibration.cams_rotation_vector[ind6][1] = rotation_vector[0][1]
        calibration.cams_rotation_vector[ind6][2] = rotation_vector[0][2]

        calibration.cams_translation[ind6][0] = cams_first_guess[ind6, 0:3, 3:4].T[0][0]
        calibration.cams_translation[ind6][1] = cams_first_guess[ind6, 0:3, 3:4].T[0][1]
        calibration.cams_translation[ind6][2] = cams_first_guess[ind6, 0:3, 3:4].T[0][2]

    for i in range(len(classe_importante.valid_collections_list)):
        chess_rotation, chess_translation = matrix_to_vector(chess_first_guess[i, :, :])
        calibration.chess_rotation_vector[i][0] = chess_rotation[0][0]
        calibration.chess_rotation_vector[i][1] = chess_rotation[0][1]
        calibration.chess_rotation_vector[i][2] = chess_rotation[0][2]

        calibration.chess_translation[i][0] = chess_translation[0]
        calibration.chess_translation[i][1] = chess_translation[1]
        calibration.chess_translation[i][2] = chess_translation[2]

    list_sensors = []

    for i_2 in sensors:
        list_sensors.append(i_2)

    ind7 = 0
    for ind8 in sensors:
        rot_vector = np.zeros(3)
        tr_vector = np.zeros(3)

        for ind9 in range(3):
            rot_vector[ind9] = calibration.cams_rotation_vector[ind7][ind9]
            tr_vector[ind9] = calibration.cams_translation[ind7][ind9]

        print("Transformation matrix from " + str(ind8) + " camera to World:")
        print(create_matrix_4by4(rot_vector, tr_vector))
        ind7 += 1

    # ---------------------------------------
    # --- Setup Optimizer
    # ---------------------------------------
    print('Initializing optimizer')
    cv2.destroyAllWindows()
    opt = OptimizationUtils.Optimizer()
    opt.addDataModel('calibration', calibration)

 # Create specialized getter and setter functions
    def setter(calibration, value, idx):
        if idx < 6 * len(sensors):
            for ind in range(len(sensors)):
                if idx == 0 + 6 * ind:
                    calibration.cams_rotation_vector[ind][0] = value[0]
                elif idx == 1 + 6 * ind:
                    calibration.cams_rotation_vector[ind][1] = value[0]
                elif idx == 2 + 6 * ind:
                    calibration.cams_rotation_vector[ind][2] = value[0]
                elif idx == 3 + 6 * ind:
                    calibration.cams_translation[ind][0] = value[0]
                elif idx == 4 + 6 * ind:
                    calibration.cams_translation[ind][1] = value[0]
                elif idx == 5 + 6 * ind:
                    calibration.cams_translation[ind][2] = value[0]
        elif 6 * len(sensors) <= idx < 6 * len(sensors) + 6 * len(list_detected):
            for ind in range(len(list_detected)):
                if idx == 6 * len(sensors) + 6 * ind:
                    calibration.chess_rotation_vector[ind][0] = value[0]
                elif idx == 6 * len(sensors) + 1 + 6 * ind:
                    calibration.chess_rotation_vector[ind][1] = value[0]
                elif idx == 6 * len(sensors) + 2 + 6 * ind:
                    calibration.chess_rotation_vector[ind][2] = value[0]
                elif idx == 6 * len(sensors) + 3 + 6 * ind:
                    calibration.chess_translation[ind][0] = value[0]
                elif idx == 6 * len(sensors) + 4 + 6 * ind:
                    calibration.chess_translation[ind][1] = value[0]
                elif idx == 6 * len(sensors) + 5 + 6 * ind:
                    calibration.chess_translation[ind][2] = value[0]
        elif 6 * len(sensors) + 6 * len(list_detected) <= idx < (9 + 6) * len(sensors) + 6 * len(list_detected):
            x = 6 * len(sensors) + 6 * len(list_detected)
            for ind in range(len(sensors)):
                if idx == x + 9 * ind:
                    calibration.sensors[ind].k[0, 0] = value[0]
                elif idx == x + 9 * ind + 1:
                    calibration.sensors[ind].k[0, 1] = value[0]
                elif idx == x + 9 * ind + 2:
                    calibration.sensors[ind].k[0, 2] = value[0]
                elif idx == x + 9 * ind + 3:
                    calibration.sensors[ind].k[1, 0] = value[0]
                elif idx == x + 9 * ind + 4:
                    calibration.sensors[ind].k[1, 1] = value[0]
                elif idx == x + 9 * ind + 5:
                    calibration.sensors[ind].k[1, 2] = value[0]
                elif idx == x + 9 * ind + 6:
                    calibration.sensors[ind].k[2, 0] = value[0]
                elif idx == x + 9 * ind + 7:
                    calibration.sensors[ind].k[2, 1] = value[0]
                elif idx == x + 9 * ind + 8:
                    calibration.sensors[ind].k[2, 2] = value[0]
        elif idx >= (9 + 6) * len(sensors) + 6 * len(list_detected):
            x = (9 + 6) * len(sensors) + 6 * len(list_detected)
            for ind in range(len(sensors)):
                if idx == x + 5 * ind:
                    calibration.sensors[ind].dist[0] = value[0]
                elif idx == x + 5 * ind + 1:
                    calibration.sensors[ind].dist[1] = value[0]
                elif idx == x + 5 * ind + 2:
                    calibration.sensors[ind].dist[2] = value[0]
                elif idx == x + 5 * ind + 3:
                    calibration.sensors[ind].dist[3] = value[0]
                elif idx == x + 5 * ind + 4:
                    calibration.sensors[ind].dist[4] = value[0]
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
        elif 6 * len(sensors) <= idx < 6 * len(sensors) + 6 * len(list_detected):
            for ind1 in range(len(list_detected)):
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
        elif 6 * len(sensors) + 6 * len(list_detected) <= idx < (9 + 6) * len(sensors) + 6 * len(list_detected):
            x = 6 * len(sensors) + 6 * len(list_detected)
            for ind in range(len(sensors)):
                if idx == x + 9 * ind:
                    return [calibration.sensors[ind].k[0, 0]]
                elif idx == x + 9 * ind + 1:
                    return [calibration.sensors[ind].k[0, 1]]
                elif idx == x + 9 * ind + 2:
                    return [calibration.sensors[ind].k[0, 2]]
                elif idx == x + 9 * ind + 3:
                    return [calibration.sensors[ind].k[1, 0]]
                elif idx == x + 9 * ind + 4:
                    return [calibration.sensors[ind].k[1, 1]]
                elif idx == x + 9 * ind + 5:
                    return [calibration.sensors[ind].k[1, 2]]
                elif idx == x + 9 * ind + 6:
                    return [calibration.sensors[ind].k[2, 0]]
                elif idx == x + 9 * ind + 7:
                    return [calibration.sensors[ind].k[2, 1]]
                elif idx == x + 9 * ind + 8:
                    return [calibration.sensors[ind].k[2, 2]]
        elif idx >= (9 + 6) * len(sensors) + 6 * len(list_detected):
            x = (9 + 6) * len(sensors) + 6 * len(list_detected)
            for ind in range(len(sensors)):
                if idx == x + 5 * ind:
                    return [calibration.sensors[ind].dist[0]]
                elif idx == x + 5 * ind + 1:
                    return [calibration.sensors[ind].dist[1]]
                elif idx == x + 5 * ind + 2:
                    return [calibration.sensors[ind].dist[2]]
                elif idx == x + 5 * ind + 3:
                    return [calibration.sensors[ind].dist[3]]
                elif idx == x + 5 * ind + 4:
                    return [calibration.sensors[ind].dist[4]]
        else:
            raise ValueError('Unknown i value: ' + str(idx))
    parameter_names = []
    for ind2 in range(len(sensors)):
        parameter_names.extend(['r1' + str(ind2), 'r2' + str(ind2), 'r3' + str(ind2), 't1' + str(ind2), 't2' + str(ind2), 't3' + str(ind2)])

    for ind2 in range(len(list_detected)):
        parameter_names.extend(['rchess1_' + str(ind2), 'rchess2_' + str(ind2), 'rchess3_' + str(ind2), 'txchess_' + str(ind2), 'tychess_' + str(ind2), 'tzchess_' + str(ind2)])

    for ind in range(len(sensors)):
        for ind1 in range(9):
            a = str('k' + str(ind1) + '_s' + str(ind))
            parameter_names.append(a)

    for ind in range(len(sensors)):
        for ind1 in range(5):
            a = str('dist' + str(ind1) + '_s' + str(ind)    )
            # print a
            # exit(0)
            # parameter_names.extend(str(a))
            parameter_names.append(a)
    print parameter_names
    # exit(0)

    for idx in range(0, 6 * (len(list_detected) + len(sensors)) + 5 * len(sensors) + 9 * len(sensors)):
        opt.pushParamScalar(group_name=parameter_names[idx], data_key='calibration', getter=partial(getter, idx=idx), setter=partial(setter, idx=idx))

   # ---------------------------------------
    # --- Define THE OBJECTIVE FUNCTION
    # ---------------------------------------
    def objectiveFunction(model):

        calibration = model['calibration']
        error = []
        calibration.resetImagePoints()

        cams_rotation_vector = np.zeros((len(sensors), 1, 3), np.float32)
        cams_translation = np.zeros((len(sensors), 3), np.float32)
        chess_rotation_vector = np.zeros((1, 3), np.float32)
        chess_translation = np.zeros((3), np.float32)
        for ind, col in enumerate(list_detected):

            for ind5 in range(0, 3):
                for i2 in range(len(sensors)):
                    cams_rotation_vector[i2, 0, ind5] = calibration.cams_rotation_vector[i2][ind5]
                    cams_translation[i2, ind5] = calibration.cams_translation[i2][ind5]
                chess_rotation_vector[0, ind5] = calibration.chess_rotation_vector[ind][ind5]
                chess_translation[ind5] = calibration.chess_translation[ind][ind5]

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

            for ind3, i in enumerate(sensors):
                if calibration_data['collections'][str(list_detected[ind])]['data'][str(i)]['detected'] == 1:

                    cams_T_world = create_matrix_4by4(cams_rotation_vector[ind3, :, :], cams_translation[ind3, :])
                    transformation_matrix = np.dot(cams_T_world, inv_chess_T_world)

                    r_cam2tochess_vector, t_cam2tochess = matrix_to_vector(transformation_matrix)

                    imgpoints_optimize = cv2.projectPoints(pts_chessboard, r_cam2tochess_vector, t_cam2tochess,
                                                                 calibration.sensors[ind3].k, calibration.sensors[ind3].dist)

                    calibration.sensors[ind3].image_points.append(imgpoints_optimize)

                    imgpoints_real = find_chess_points(str(calibration.sensors[ind3].image + str(col) + '.jpg'))

                    for a in range(dimensions[0] * dimensions[1]):

                        error.append(
                            math.sqrt((imgpoints_optimize[0][a][0][0] - imgpoints_real[0][a][0][0]) ** 2 + (
                                imgpoints_optimize[0][a][0][1] - imgpoints_real[0][a][0][1]) ** 2))

        # print("avg error: " + str(np.mean(np.array(error))))
        # error = [0 if a_ > 1e15 else a_ for a_ in error]
        return error

    opt.setObjectiveFunction(objectiveFunction)

    # ---------------------------------------
    # --- Define THE RESIDUALS
    # ---------------------------------------

    for a1 in list_detected:
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
        #
        # for index, col in enumerate(calibration.collections):
        #     for i, s in enumerate(sensors):
        #         index_sensor = calibration.sensors[i].image_index
        #
        #         img1 = cv2.imread(calibration.sensors[i].image + str(col) + '.jpg')
        #         if classe_importante.filtered_matrix[index, i] == 1:
        #             image_points = calibration.sensors[i].image_points[index_sensor][0]
        #             img1 = cv2.drawChessboardCorners(img1, (9, 6), image_points, True)
        #             print image_points
        #             # if index_sensor == len(calibration.sensors[i].image_points):
        #             #     calibration.sensors[i].image_index = 0
        #             # else:
        #             calibration.sensors[i].image_index += 1
        #         cv2.imshow('img_' + s, img1)
        #     cv2.waitKey(1000)
        #
        # #
        # half = math.ceil(len(classe_importante.sensors_list)/2)
        # for index, col in enumerate(calibration.collections):
        #     for i, s in enumerate(sensors):
        #         index_sensor = calibration.sensors[i].image_index
        #         if i == 0:
        #             # print(len(calibration.sensors[i].image_points))
        #             # print(len(calibration.collections))
        #             # exit(0)
        #             img1 = cv2.imread(calibration.sensors[i].image + str(col) + '.jpg')
        #             if classe_importante.filtered_matrix[index, i] == 1:
        #                 img1 = cv2.drawChessboardCorners(img1, (9, 6), calibration.sensors[i].image_points[index_sensor][0], True)
        #                 if index_sensor == len(calibration.sensors[i].image_points):
        #                     print(index_sensor)
        #                     exit(0)
        #                     calibration.sensors[i].image_index = 0
        #                 else:
        #                     calibration.sensors[i].image_index += 1
        #             continue
        #         if i == half:
        #             img2 = cv2.imread(calibration.sensors[i].image + str(col) + '.jpg')
        #             if classe_importante.filtered_matrix[index, i] == 1:
        #                 img2 = cv2.drawChessboardCorners(img2, (9, 6), calibration.sensors[i].image_points[index_sensor][0], True)
        #                 if index_sensor == len(calibration.sensors[i].image_points):
        #                     calibration.sensors[i].image_index = 0
        #                 else:
        #                     calibration.sensors[i].image_index += 1
        #             continue
        #         if i < half:
        #             img = cv2.imread(calibration.sensors[i].image + str(col) + '.jpg')
        #             if classe_importante.filtered_matrix[index, i] == 1:
        #                 img = cv2.drawChessboardCorners(img, (9, 6), calibration.sensors[i].image_points[index_sensor][0], True)
        #                 if index_sensor == len(calibration.sensors[i].image_points):
        #                     calibration.sensors[i].image_index = 0
        #                 else:
        #                     calibration.sensors[i].image_index += 1
        #             img1 = cv2.hconcat((img1, img))
        #         else:
        #             img = cv2.imread(calibration.sensors[i].image + str(col) + '.jpg')
        #             if classe_importante.filtered_matrix[index, i] == 1:
        #                 img = cv2.drawChessboardCorners(img, (9, 6), calibration.sensors[i].image_points[index_sensor][0], True)
        #                 if index_sensor == len(calibration.sensors[i].image_points):
        #                     calibration.sensors[i].image_index = 0
        #                 else:
        #                     calibration.sensors[i].image_index += 1
        #             img2 = cv2.hconcat((img2, img))
        #     # img_final = cv2.vconcat  ((img1, img2))
        #     cv2.imshow('img1', img1)
        #     cv2.imshow('img2', img2)
        #     cv2.waitKey(1000)
        # wm = KeyPressManager.KeyPressManager.WindowManager()
        # img4 = cv2.imread(name_image_left[ind1])
        # img5 = cv2.drawChessboardCorners(img4, (9, 6), calibration.left_cam_image_points[ind1][0], True)
        # cv2.imshow('img_left', img5)
        # cv2.waitKey(1)
        # wm = KeyPressManager.KeyPressManager.WindowManager()
        pass


    opt.setVisualizationFunction(visualizationFunction, True)

    print("\n\nStarting optimization")
    opt.startOptimization(
        optimization_options={'x_scale': 'jac', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'diff_step': 1e-4})

    for ind7, ind8 in enumerate(sensors):
        rot_vector = np.zeros(3)
        tr_vector = np.zeros(3)

        for ind9 in range(3):
            rot_vector[ind9] = calibration.cams_rotation_vector[ind7][ind9]
            tr_vector[ind9] = calibration.cams_translation[ind7][ind9]

        print("First Guess from " + str(ind8) + " camera to World:")
        print(cams_first_guess[ind7, :, :])
        print("Transformation matrix from " + str(ind8) + " camera to World:")
        print(create_matrix_4by4(rot_vector, tr_vector))

    erro = objectiveFunction(calibration)
    print("################################\n")
    print ("Erro medio Final: ")
    print (np.mean(erro))
    print("################################\n")
    print("################################\n")
    print("Numero total de residuais: ")
    print(len(erro))
    print("################################\n")

    calibration.showResults(classe_importante.sensors_first_guess)
