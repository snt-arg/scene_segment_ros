#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError
from utils.helpers import cleanMemory, monitorParams


class Segmenter:
    def __init__(self):
        # Initial checks
        monitorParams()
        cleanMemory()

        # Get parameters
        print('Loading configuration parameters ...\n')

    def segmentation(self, imageMessage):
        try:
            print('Here')

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


# Run the program
if __name__ == '__main__':
    # Initialization
    rospy.init_node('segmenter', anonymous=False)
    segmenter = Segmenter()
    rospy.spin()
