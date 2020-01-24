from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import imutils
import cv2
from imutils import paths

def predictImage(model, filePath, modeltype):
    image = cv2.imread(filePath)
    w=32
    h=32
    output = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (w, h))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image

    (Outlier,Normal,Tumor) = model.predict(image)[0]

    # build the label

    label = "{:.2f}%, {:.2f}%, {:.2f}%".format(Outlier * 100,Normal * 100,Tumor * 100)
    # draw the label on the image
##    output = imutils.resize(image, width=400)
    
    if modeltype=="CNN":        
        cv2.putText(output, "CNN prediction:", (10, 20),  cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
        cv2.putText(output, label, (10, 40),  cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 32, 255), 2)
    else:
        cv2.putText(output, "CapsNet prediction", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
        cv2.putText(output, label, (10, 80),  cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 32, 255), 2)
        
    values = (Outlier,Normal,Tumor)
    maxProb = values.index(max(values))
    color = (0,32,255)
    classType = 0
    if maxProb== 1:
            color = (0,255,0)
            classType = 1
    if maxProb== 2:
            color = (0,0,0)
            classType = 2
    cv2.rectangle(output,(0,0),(np.size(output, 1),np.size(output, 0)),color,5)
    cv2.imwrite(filePath,output)
    return classType 
