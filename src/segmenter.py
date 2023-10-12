#!/usr/bin/env python

import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from utils.helpers import cleanMemory, monitorParams


class Segmenter:
    def __init__(self):
        # Initial checks
        monitorParams()
        cleanMemory()

        # Get parameters
        print('Loading configuration parameters ...')
        params = rospy.get_param('~params')
        modelName = params['model_params']['model_name']
        modelPath = params['model_params']['model_path']

        # Subscribers
        # rospy.Subscriber(pose_topic, PoseStamped, self.sub_pose)

        # Publishers
        # publisherMask = rospy.Publisher('result_mask', Image, queue_size=10)

        # Read input images
        # image = cv.imread(imagePath)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # # Run Segmentation
        # fsam = fastSamInit()
        # masks = fastSamSegmenter(image, fsam)
        # fastSamShowOutput(masks)


# Run the program
if __name__ == '__main__':
    # Initialization
    rospy.init_node('segmenter', anonymous=False)
    segmenter = Segmenter()
    rospy.spin()
