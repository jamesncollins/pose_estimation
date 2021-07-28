from cv2 import cv2
import sys, os, requests 

#Skeleton
ALL_JOINTS = ['Head', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
              'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle',
              'LHip', 'LKnee', 'LAnkle', 'Chest', 'Background']
EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
         [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

PROTO_FILE = "content/pose_deploy_linevec_faster_4_stages.prototxt.txt"
WEIGHTS_FILE = "content/pose_iter_160000.caffemodel"

THRESHOLD = 0.3

SCALE_FACTOR = 1.0 / 255
IN_HEIGHT = 368
IN_WIDTH = 368
RGB_MEAN = (0, 0, 0)

JOINT_DOT_WIDTH = 15
JOINT_DOT_COLOR = (0, 0, 255)
JOINT_TEXT_COLOR = (0, 255, 0)
JOINT_LINE_COLOR = (0, 255, 255)
JOINT_LINE_THICKNESS = 3
JOINT_FONT_SCALE = 0.75
JOINT_FONT_THICKNESS = 2

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

class Vectoriser():
    """
    Calculates and stores the vector representing the pose.  
    """
    def get_vector(self, skeleton):
        print(skeleton)
        
class SkeletonSketch():

    def sketch(self, points, file_name):
        """
        Draw points and connect with lines on an image. 
        """
        self.points = points
        self.img = cv2.imread(file_name)
        self.__sketch_dots()
        self.__sketch_lines()
        return self.img

    def __sketch_dots(self):
        #draw points and labels.
        for i, joint in enumerate(ALL_JOINTS): 
            if self.points[i] != None:
                x = self.points[i][0]
                y = self.points[i][1]
                cv2.circle(self.img, (x,y), JOINT_DOT_WIDTH, JOINT_DOT_COLOR, thickness=-1, lineType=cv2.FILLED)
                cv2.putText(self.img, "{}".format(joint), (x,y), cv2.FONT_HERSHEY_COMPLEX, JOINT_FONT_SCALE, JOINT_TEXT_COLOR, JOINT_FONT_THICKNESS, lineType=cv2.LINE_AA)

    def __sketch_lines(self):
        #draw lines between points. 
        for pair in EDGES:
            point_a = self.points[pair[0]]
            point_b = self.points[pair[1]]

            if not point_a is None and not point_b is None:
                cv2.line(self.img, point_a, point_b, JOINT_LINE_COLOR, JOINT_LINE_THICKNESS)
 

class Estimator():
    """
    Responsible for calculating the pose from an image and storing the points. 
    """
    def __init__(self):
        #Downloads the pre-trained model if not already present. 
        if not os.path.isfile(WEIGHTS_FILE):
            print("Retrieving model from 'posefs1.perception.cs.cmu.edu'. This will take a few minutes. Subsequent runs will skip this step.")
            url = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel'
            r = requests.get(url, allow_redirects=True)
            open(WEIGHTS_FILE, 'wb').write(r.content)
    
    def skeleton(self, file_path):
        #run network on image.
        img = cv2.imread(file_path)
        output = self.__learn_image(img)

        #get image points to plot.  
        return self.__get_image_points(output, img)

    def __learn_image(self, img):
        """
        Runs the DNN on the input image.  
        """
        #load the model weights onto the network
        net = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)

        #input the image into the network
        inp_blob = cv2.dnn.blobFromImage(img, SCALE_FACTOR, (IN_WIDTH, IN_HEIGHT), RGB_MEAN, swapRB=False, crop=False)
        net.setInput(inp_blob)
        return net.forward()
    
    def __get_image_points(self, output, img):
        """
        Extract the joint positions and map to image dimensions. 
        """
        rows = output.shape[2]
        cols = output.shape[3]

        points = []
        for i in range(len(ALL_JOINTS)):
            #get heat map for each joint 
            prob_map = output[0, i, :, :]

            #extract predicted region for each joint 
            _, prob, _, point = cv2.minMaxLoc(prob_map) 

            #map region of output to point on image. 
            x = int((img.shape[1] * point[0]) / rows)
            y = int((img.shape[0] * point[1]) / cols)

            if prob > THRESHOLD:
                points.append((x,y))
            else:
                points.append(None)
        return points 