#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pickle
from imutils import paths
import random
import cv2
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split


TRAIN_FILE = "train.p"
VALID_FILE = "valid.p"
TEST_FILE = "test.p"


def get_data(folder,NUM_LABEL):
    

    # initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(folder)))
    random.seed(NUM_LABEL)
    random.shuffle(imagePaths)

    w = 32
    h = 32
    
    sub_dirs = [x[0] for x in os.walk(folder)]

    # The root directory comes first, so skip it.

    is_root_dir = True
    for sub_dir in sub_dirs:
            if is_root_dir:
                    is_root_dir = False
                    continue
            dir_name = os.path.basename(sub_dir)

    num_classes = len(dir_name)

    # loop over the input images
    for imagePath in imagePaths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (w, h))
            image = img_to_array(image)
            data.append(image)

            # extract the class label from the image path and update the
            # labels list
            label = imagePath.split(os.path.sep)[-2]

            if label == "Outlier":
                    label = 0 
            if label == "Normal":
                    label = 1 
            if label == "Tumor":
                    label = 2
            labels.append(label)


    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float")
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (x, X_test, y, y_test) = train_test_split(data,
            labels, test_size=0.25, random_state=NUM_LABEL)

    (X_train, X_valid, y_train,  y_valid) = train_test_split(x,
            y, test_size=0.25, random_state=NUM_LABEL)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test
