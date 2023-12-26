# import the opencv library
import cv2
import os
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
ground_t = '/Standstill'
path = 'C:/Users/aksha/OneDrive/Desktop/Desktop/Spring 2023/Estimation Detection and Learning/Final Project/Test_Data' + str(ground_t)
sub_list = os.listdir(path)
sub_list = [int(i) for i in sub_list]
# Create new test folder if it does not exist
if len(sub_list) == 0:
    new_dir_path = os.path.join(path, str(1))
    os.mkdir(new_dir_path)
else:
    new_folder = max(sub_list) + 1
    new_dir_path = os.path.join(path, str(new_folder))
    os.mkdir(new_dir_path)
# Change directory to new test folder to save image frames in
count = 0  
os.chdir(new_dir_path)
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imwrite('Frame'+str(count)+'.jpg', frame)
    count = count + 1
    
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()