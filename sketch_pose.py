import sys
import os
from cv2 import cv2
import pose_estimator

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FILE_NAME = sys.argv[1]
    else:
        FILE_NAME = input("Input the file name of the image to analyse: ")

    #Check that the specified image exists.
    file_path = 'content/' + FILE_NAME
    while not os.path.isfile(file_path):
        print("No file named", file_path, "exists. Make sure it's spelled correctly and in 'content' folder.")
        FILE_NAME = input("Input the file name of the image to analyse: ")

        #Check that the specified image exists.
        file_path = 'content/' + FILE_NAME
    
    #get points
    estimator = pose_estimator.Estimator()
    skeleton = estimator.skeleton(file_path)

    #put points as shapes on image
    sketchpad = pose_estimator.SkeletonSketch()
    img = sketchpad.sketch(skeleton, file_path)

    pose_estimator.display_image(img)
