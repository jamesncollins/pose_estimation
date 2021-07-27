pose_estimation
==============

A python library for tools estimating a person's from pose from a picture. 

To use: 

1. Download the github directory

```
git clone https://github.com/jamesncollins/pose_estimation.git
```

2. Place the image you want to analyse in the '/content' directory. 

3. Navigate to the 'pose_estimation' directory and run:

```
python sketch_pose.py <IMAGE-FILE-NAME>
```

![Katharine](https://github.com/jamesncollins/pose_estimation/blob/content/katharine.png?raw=true)



## Notes

The pose estimation and visualisation uses a pre-trained Caffe model from OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose), following the implementation from Learn OpenCV available here (https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/)

Make sure to include the full image file name including extension. 

The tool works best when: 
- the picture only has one person
- the picture is not grainy
- the person is shown from at least waist up
- the person is not obscuring their body e.g., crouching, squatting, wearing baggy clothes. 

