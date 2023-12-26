import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Function reads image and binarizes it
def get_image_binarize(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# Function gets disparity map by comparing every 5 images
def get_disparity_map_test(path, class_type):
    # Get correct directory
    dir_list = os.listdir(path)
    dir_list = np.array(dir_list)
    class_dir = 'C:/Users/aksha/OneDrive/Desktop/Desktop/Spring 2023/Estimation Detection and Learning/Final Project/Disparity_Map'
    class_dir = class_dir + '/' + class_type + '/disp'
    count = 0
    # Loop through each test case
    for i in range(len(dir_list)-5):
        # Get image path for every 5th image and compare
        frame_num_1 = dir_list[i]
        frame_num_2 = dir_list[i+5]
        path_1 = path + '/' + frame_num_1
        path_2 = path + '/' + frame_num_2
        gray_1 = get_image_binarize(path_1)
        gray_2 = get_image_binarize(path_2)
        # Compare grayscale images between every 5th image and disparity map
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)
        # Save disparity map
        disparity = stereo.compute(gray_1,gray_2)
        disp_path = class_dir+str(count)+'.jpg'
        cv2.imwrite(disp_path, disparity)
        count = count + 1



# Loop thrugh collected test directory to get all test examples
test_dir = 'C:/Users/aksha/OneDrive/Desktop/Desktop/Spring 2023/Estimation Detection and Learning/Final Project/Test_Data'
test_list = os.listdir(test_dir)
# Loop through all classes
for classes in test_list:
    class_path = test_dir + '/' + classes
    num_test_list = os.listdir(class_path)
    # Loop through all tests in each class
    for test_num in num_test_list:
        test_num_path = class_path + '/' + test_num
        # Compute disparity map for each test
        get_disparity_map_test(test_num_path, classes)

