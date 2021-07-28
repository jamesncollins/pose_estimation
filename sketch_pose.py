from cv2 import cv2
import sys # os, requests 
import pose_estimator

def display_image(img):
    """
    Display image. 
    """       
    #display the image with the OpenCV image display. 
    cv2.imshow('image', img)

    #press ESC to exit display. 
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows() 

if __name__ == "__main__":
    FILE_NAME = sys.argv[1]

    estimator = pose_estimator.Estimator()
    skeleton = estimator.skeleton(FILE_NAME)

    sketchpad = pose_estimator.SkeletonSketch()
    img = sketchpad.sketch(skeleton, FILE_NAME)

    display_image(img)