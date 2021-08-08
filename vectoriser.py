from cv2 import cv2
import sys
import os 
import requests 
import pose_estimator
from scipy import spatial

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

    #get utility objects
    estimator = pose_estimator.Estimator()
    vectoriser = pose_estimator.Vectoriser()

    #get points of image
    skeleton = estimator.skeleton(file_path)

    #convert points into vector 
    vector = vectoriser.get_slants(skeleton)

    print("Type in a number to select an option.")
    print("1) Print the angles representing the pose.")
    print("2) Compare the pose to a image from another image.")
    print("3) Exit program")
    selection = int(input("Type in your selection: "))

    while selection not in [1,2,3]:
        print("That's not one of the options. Type in 1, 2, or 3.")
        print("1) Print the angles representing the pose.")
        print("2) Compare the pose to a image from another image.")
        print("3) Exit program")
        selection = int(input("Type in your selection: "))

    if selection == 1:
        print(vector)

    elif selection == 2:
        #get comparison image
        COMP_NAME = input("Type in the file name of the image to compare with " + FILE_NAME + ": ")
        comp_path = 'content/' + COMP_NAME

        #check file exists
        while not os.path.isfile(comp_path):
            print("No file named", comp_path, "exists. Make sure it's spelled correctly and in 'content' folder.")
            COMP_NAME = input("Type in the file name of the image to compare with " + FILE_NAME + ": ")
            comp_path = 'content/' + COMP_NAME
        
        #compare with image:
        comp_skeleton = estimator.skeleton(comp_path)
        comp_vector = vectoriser.get_slants(comp_skeleton)
        print(1 - spatial.distance.cosine(comp_vector, vector)
)



        

