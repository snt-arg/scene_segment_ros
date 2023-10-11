#!/usr/bin/env python

import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class Segmenter:
    def __init__(self):
        # Get parameters
        print('Loading config parameters ...')
        params = rospy.get_param('~params')
        modelName = params['modelName']

        # Subscribers
        # rospy.Subscriber(pose_topic, PoseStamped, self.sub_pose)

        # Publishers
        # publisherMask = rospy.Publisher('result_mask', Image, queue_size=10)


# Run the program
if __name__ == '__main__':
    # Initialization
    rospy.init_node('segmenter', anonymous=False)
    segmenter = Segmenter()
    rospy.spin()
