import sys
import os
from cv2 import cv2
import pose_estimator

if __name__ == "__main__":
    FILE_NAME = sys.argv[1]

    #Check that the specified image exists.
    file_path = 'content/' + FILE_NAME
    if os.path.isfile(file_path):

        #get points
        estimator = pose_estimator.Estimator()
        skeleton = estimator.skeleton(file_path)

        sketchpad = pose_estimator.SkeletonSketch()
        img = sketchpad.sketch(skeleton, file_path)

        pose_estimator.display_image(img)

    else:
        print("No file named", file_path, "exists. Make sure it's spelled correctly and in 'content' folder.")