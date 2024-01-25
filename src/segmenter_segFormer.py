#!/usr/bin/env python

import time
import rospy
from output import SegFormerVisualizer, SegFormerEntropyVisualizer
from sensor_msgs.msg import Image
from modelRunner import SegFormerSegmenter, SegFormerInit
from cv_bridge import CvBridge, CvBridgeError
from utils.semantic_utils import cleanMemory, monitorParams


class Segmenter:
    def __init__(self):
        # Initial checks
        monitorParams()
        cleanMemory()

        # Get parameters
        print('Loading configuration parameters ...\n')
        params = rospy.get_param('~params')
        self.conf = params['model_params']['conf']
        modelName = params['model_params']['model_name']
        self.imageSize = params['image_params']['image_size']
        rawImageTopic = params['ros_topics']['raw_image_topic']
        segImageTopic = params['ros_topics']['segmented_image_topic']

        # Initial the segmentation module
        self.model, self.image_processor = SegFormerInit(modelName)

        # Subscribers
        rospy.Subscriber(rawImageTopic, Image, self.segmentation)

        # Publishers
        self.publisherSeg = rospy.Publisher(
            segImageTopic, Image, queue_size=10)
        self.publisherUnc = rospy.Publisher(
            segImageTopic + "/uncertainty", Image, queue_size=10)

        # ROS Bridge
        self.bridge = CvBridge()

    def segmentation(self, imageMessage):
        try:
            # Convert the ROS Image message to a CV2 image
            cvImage = self.bridge.imgmsg_to_cv2(imageMessage, "bgr8")

            # Processing
            predictions = SegFormerSegmenter(cvImage, self.model, self.image_processor)
            segmentedImage = SegFormerVisualizer(cvImage, predictions)
            segmentedEntropyImage = SegFormerEntropyVisualizer(cvImage, predictions)

            # Publish the processed image
            processedImgMsg = self.bridge.cv2_to_imgmsg(
                segmentedImage, "bgr8")
            self.publisherSeg.publish(processedImgMsg)

            processedEntropyImgMsg = self.bridge.cv2_to_imgmsg(
                segmentedEntropyImage, "bgr8")
            self.publisherUnc.publish(processedEntropyImgMsg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


# Run the program
if __name__ == '__main__':
    # Initialization
    rospy.init_node('segmenter', anonymous=False)
    segmenter = Segmenter()
    rospy.spin()
