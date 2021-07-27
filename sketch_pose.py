from cv2 import cv2
import sys, os, requests 

PART_DICT = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
"LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
"RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
"Background": 15}
NPOINTS = len(PART_DICT)

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
             ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
             ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
             ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

PROTO_FILE = "content/pose_deploy_linevec_faster_4_stages.prototxt.txt"
WEIGHTS_FILE = "content/pose_iter_160000.caffemodel"

THRESHOLD = 0.3

SCALE_FACTOR = 1.0 / 255
IN_HEIGHT = 368
IN_WIDTH = 368
RGB_MEAN = (0, 0, 0)

JOINT_DOT_WIDTH = 15
JOINT_DOT_COLOR = (0, 255, 255)
JOINT_TEXT_COLOR = (0, 0, 255)
JOINT_LINE_COLOR = (0, 255, 0)
JOINT_LINE_THICKNESS = 3
JOINT_FONT_SCALE = 1.4
JOINT_FONT_THICKNESS = 3

class Estimator():
    def __init__(self, file_name):
        self.check_weights()
        file_path = 'content/' + file_name
        if os.path.isfile(file_path):
            output, img = self.learn_image(file_path)
            self.print_pose(output, img)
        else:
            print("No file named", file_name, "exists. Make sure it's spelled correctly and in 'Content'")

    def check_weights(self):
        if os.path.isfile(WEIGHTS_FILE):
            print("Model available.")
            return 
        else:
            print("Retrieving model from 'posefs1.perception.cs.cmu.edu'. This will take a few minutes. Subsequent runs will skip this step.")
            url = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel'
            r = requests.get(url, allow_redirects=True)
            open(WEIGHTS_FILE, 'wb').write(r.content)

    def learn_image(self, file_path):
        net = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)
        img = cv2.imread(file_path)
        inp_blob = cv2.dnn.blobFromImage(img, SCALE_FACTOR, (IN_WIDTH, IN_HEIGHT), RGB_MEAN, swapRB=False, crop=False)
        net.setInput(inp_blob)
        return net.forward(), img
    
    def print_pose(self, output, img):
        W = output.shape[2]
        H = output.shape[3]

        points = []

        for i in range(NPOINTS):
            prob_map = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(prob_map)

            x = int((img.shape[1] * point[0]) / W)
            y = int((img.shape[0] * point[1]) / H)

            if prob > THRESHOLD:
                cv2.circle(img, (x,y), JOINT_DOT_WIDTH, JOINT_DOT_COLOR, thickness=-1, lineType=cv2.FILLED)
                cv2.putText(img, "{}".format(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, JOINT_FONT_SCALE, JOINT_TEXT_COLOR, JOINT_FONT_THICKNESS, lineType=cv2.LINE_AA)
                points.append((x,y))
            else:
                points.append(None)
        
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            pointA = points[PART_DICT[partA]]
            pointB = points[PART_DICT[partB]]

            if not pointA is None and not pointB is None:
                cv2.line(img, pointA, pointB, JOINT_LINE_COLOR, JOINT_LINE_THICKNESS)
        
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows() 

if __name__ == "__main__":
    file_name = sys.argv[1]
    Estimator(file_name)