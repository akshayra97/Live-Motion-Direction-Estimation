from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score)
import matplotlib.pyplot as plt
import skimage
import numpy as np
import os
import cv2

## Image preparation
label_list = ['up','down','left','right','forward','backward']
parent_fldr = os.path.dirname(os.path.realpath(__file__))       # Get folder where this script is
image_fldrs = os.listdir(parent_fldr + "/Images")               # Get all of the folders containing images
image_fldrs = [x.lower() for x in image_fldrs]                  # Make lowercase to match label_list

dispars = []
labels = []

for folder in image_fldrs:
    images = os.listdir(parent_fldr + "/Images/" + folder) # List of all images in that folder
    txt = "Processing {curr_folder}"
    num_img = 0
    print(txt.format(curr_folder = folder))
    for image in images:
        num_img += 1
        if num_img > 300:
            break

        # Get the full path to one image
        full_path = parent_fldr + "/Images/" + folder + "/" + image
        
        # Load image
        img = cv2.imread(full_path)

        # Image should already be 2D grayscale for disparity map, but let's make sure
        if(len(img.shape)>2):
            img = img[:,:,0]

        # Down sample with max pooling to reduce image size
        img = skimage.measure.block_reduce(img,block_size = 2, func = np.max)

        h = img.shape[0] # height
        w = img.shape[1] # width
        img = img.reshape([1,h*w]) # Make into a single row

        # Stack each disparity map to the full list
        dispars.append(img)

        # Add the images label to the full list. The label is taken from the folder it's in
        labels.append(label_list.index(folder))

dispars = np.concatenate(dispars) # Convert the list of arrays into a single 2D array

## Model Creation and Application
X_train, X_test, y_train, y_test = train_test_split(dispars, labels, test_size=0.2, random_state=125)

# Use support vector classification
clf = SVC()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)
print("Prediction accuracy:",accuracy*100,"%")

