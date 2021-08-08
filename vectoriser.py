from cv2 import cv2
import sys
import os 
import requests 
import pose_estimator

if __name__ == "__main__":
    FILE_NAME = sys.argv[1]

    #Check that the specified image exists.
    file_path = 'content/' + FILE_NAME
    if os.path.isfile(file_path):

        #get points
        estimator = pose_estimator.Estimator()
        skeleton = estimator.skeleton(file_path)

        #convert points into vector 
        vectoriser = pose_estimator.Vectoriser()
        vectoriser.get_slants(skeleton)

    else:
        print("No file named", FILE_NAME, "exists. Make sure it's spelled correctly and in 'content' folder.")