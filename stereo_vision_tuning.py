import cv2
import numpy as np
import argparse
import pathlib
import os
import matplotlib.pyplot as plt

def nothing(x):
    pass

path_img_left = r"C:\Users\aksha\OneDrive\Desktop\Desktop\Spring 2023\Estimation Detection and Learning\Final Project\Test_Data\Forward\3\Frame60.jpg"
path_img_right = r"C:\Users\aksha\OneDrive\Desktop\Desktop\Spring 2023\Estimation Detection and Learning\Final Project\Test_Data\Forward\3\Frame65.jpg"
img_left = cv2.imread(path_img_left)
img_right = cv2.imread(path_img_right)
# Show the two input images
cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
cv2.imshow("Left",img_left)
cv2.imshow("Right",img_right)

cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
 
cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
satisfied = False
plt.figure()
print("Press ESC once you have selected parameter values.\n")
while True:
    # Updating the parameters based on the trackbar positions
    numDisparity = cv2.getTrackbarPos('numDisparities','disp')*16
    block = cv2.getTrackbarPos('blockSize','disp')

    stereo = cv2.StereoSGBM_create(numDisparities=numDisparity,blockSize=block)
    disparity = stereo.compute(img_left, img_right)
    disp_full = (disparity-disparity.min())/(disparity.max()-disparity.min())*255

    cv2.imshow("disp", disparity)

    # Close window using esc key
    if cv2.waitKey(1) == 27:
        # save the disparity image and continue
        break
