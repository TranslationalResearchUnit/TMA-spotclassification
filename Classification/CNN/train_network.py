

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks  import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from modelCNN.lenet import LeNet
from modelCNN import resnet
from modelCNN.smallervggnet import SmallerVGGNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-o", "--output",  type=str, default=".\\",
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 1000
INIT_LR = 1e-3
BS = 32
w =32
h=32


# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)




sub_dirs = [x[0] for x in os.walk(args["dataset"])]

# The root directory comes first, so skip it.
dir_name = []
is_root_dir = True
for sub_dir in sub_dirs:
        if is_root_dir:
                is_root_dir = False
                continue
        dir_name.append(os.path.basename(sub_dir))
print(dir_name)
num_classes = len(dir_name)
print(num_classes)
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
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes)
testY = to_categorical(testY, num_classes)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


                
# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=w, height=h, depth=3, classes=num_classes)
#model = SmallerVGGNet.build(width=w, height=h, depth=3,classes=num_classes)
#model = resnet.ResnetBuilder.build_resnet_18((3, w, h), num_classes)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

outFolder = args["output"]         
try:
        os.stat(outFolder)
except:
        os.mkdir(outFolder)
        
# train the network
print("[INFO] training network...")
# checkpoint
filepath=outFolder+"/Lenet_weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, callbacks=callbacks_list, verbose=1)




