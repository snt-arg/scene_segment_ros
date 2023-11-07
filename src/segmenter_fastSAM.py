#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from output import fastSamVisualizer
from cv_bridge import CvBridge, CvBridgeError
from utils.helpers import cleanMemory, monitorParams
from modelRunner import fastSamInit, fastSamSegmenter


class Segmenter:
    def __init__(self):
        # Initial checks
        monitorParams()
        cleanMemory()

        # Get parameters
        print('Loading configuration parameters ...\n')
        params = rospy.get_param('~params')
        self.iou = params['model_params']['iou']
        self.conf = params['model_params']['conf']
        modelName = params['model_params']['model_name']
        modelPath = params['model_params']['model_path']
        self.imageSize = params['image_params']['image_size']
        rawImageTopic = params['ros_topics']['raw_image_topic']
        segImageTopic = params['ros_topics']['segmented_image_topic']
        self.pointPrompt = params['model_params']['point_prompt']
        self.boxPrompt = params['model_params']['box_prompt']
        self.pointLabel = params['model_params']['point_label']
        self.counters = params['model_params']['contour']

        # Initial the segmentation module
        self.fsam = fastSamInit(modelName, modelPath)

        # Subscribers
        rospy.Subscriber(rawImageTopic, Image, self.segmentation)

        # Publishers
        self.publisherSeg = rospy.Publisher(
            segImageTopic, Image, queue_size=10)

        # ROS Bridge
        self.bridge = CvBridge()

    def segmentation(self, imageMessage):
        try:
            # Convert the ROS Image message to a CV2 image
            cvImage = self.bridge.imgmsg_to_cv2(imageMessage, "bgr8")

            # Processing
            masks = fastSamSegmenter(
                cvImage, self.fsam, self.imageSize, self.conf, self.iou)
            segmentedImage = fastSamVisualizer(
                masks, self.pointPrompt, self.boxPrompt, self.pointLabel, self.counters)

            # Publish the processed image
            processedImgMsg = self.bridge.cv2_to_imgmsg(
                segmentedImage, "bgr8")
            self.publisherSeg.publish(processedImgMsg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


# Run the program
if __name__ == '__main__':
    # Initialization
    rospy.init_node('segmenter', anonymous=False)
    segmenter = Segmenter()
    rospy.spin()
