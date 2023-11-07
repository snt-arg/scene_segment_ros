#!/usr/bin/env python

import rospy
from output import FCNVisualizer
from sensor_msgs.msg import Image
from modelRunner import FCNSegmenter, FCNInit
from cv_bridge import CvBridge, CvBridgeError
from utils.helpers import cleanMemory, monitorParams


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
        modelPath = params['model_params']['model_path']
        modelConfig = params['model_params']['model_config']
        self.imageSize = params['image_params']['image_size']
        rawImageTopic = params['ros_topics']['raw_image_topic']
        segImageTopic = params['ros_topics']['segmented_image_topic']

        # Initial the segmentation module
        self.model, self.cfg = FCNInit(modelName, modelPath, modelConfig)

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
            predictions = FCNSegmenter(cvImage, self.model)
            segmentedImage = FCNVisualizer(cvImage, predictions, self.cfg)

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
