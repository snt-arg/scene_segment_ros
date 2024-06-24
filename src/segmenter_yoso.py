#!/usr/bin/env python

import torch
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from modelRunner import yosoInit, yosoSegmenter
from utils.helpers import cleanMemory, monitorParams
from utils.semantic_utils import probabilities2ROSMsg
from output import yosoVisualizer, pFCNEntropyVisualizer
from segmenter_ros.msg import SegmenterDataMsg, VSGraphDataMsg


class Segmenter:
    def __init__(self):
        # Initial checks
        monitorParams()
        cleanMemory()

        # Get parameters
        print('Loading configuration parameters ...\n')
        params = rospy.get_param('~params')
        self.classes = params['output']['classes']
        self.conf = params['model_params']['conf']
        self.overlap = params['model_params']['overlap']
        modelName = params['model_params']['model_name']
        modelPath = params['model_params']['model_path']
        modelConfig = params['model_params']['model_config']
        self.imageSize = params['image_params']['image_size']
        rawImageTopic = params['ros_topics']['raw_image_topic']
        segImageTopic = params['ros_topics']['segmented_image_topic']
        segImageVisTopic = params['ros_topics']['segmented_image_vis']

        # Initial the segmentation module
        self.model, self.cfg = yosoInit(
            modelName, modelPath, modelConfig, self.conf, self.overlap)

        # Subscribers (to vS-Graphs)
        rospy.Subscriber(rawImageTopic, VSGraphDataMsg, self.segmentation)

        # Publishers (for vS-Graphs)
        self.publisherSeg = rospy.Publisher(
            segImageTopic, SegmenterDataMsg, queue_size=10)
        self.publisherSegVis = rospy.Publisher(
            segImageVisTopic, Image, queue_size=10)

        # ROS Bridge
        self.bridge = CvBridge()

    def segmentation(self, imageMessage):
        try:
            # Parse the input data
            keyFrameId = imageMessage.keyFrameId
            keyFrameImage = imageMessage.keyFrameImage

            # Convert the ROS Image message to a CV2 image
            cvImage = self.bridge.imgmsg_to_cv2(keyFrameImage, "bgr8")

            # Processing
            predictions, seg_dict, predictionProbs = yosoSegmenter(
                cvImage, self.model, self.classes)
            segmentedImage = yosoVisualizer(cvImage, predictions, self.cfg)
            # segmentedUncImage = pFCNEntropyVisualizer(
            #     predictionProbs)

            # Convert to ROS message
            pcdProbabilities = probabilities2ROSMsg(predictionProbs,
                                                    imageMessage.header.stamp, imageMessage.header.frame_id)

            # Create a header with the current time
            header = Header()
            header.stamp = rospy.Time.now()

            # Publish the processed image to vS-Graphs
            segmenterData = SegmenterDataMsg()
            segmenterData.header = header
            segmenterData.keyFrameId = keyFrameId
            segmenterData.segmentedImage = self.bridge.cv2_to_imgmsg(
                segmentedImage, "bgr8")
            # segmenterData.segmentedImageUncertainty = self.bridge.cv2_to_imgmsg(
            #     segmentedUncImage, "bgr8")
            # labels = torch.argmax(predictions["panoptic_seg"], axis=0)
            # unique_classes = torch.unique(labels)
            self.publisherSeg.publish(segmenterData)

            # Publish the processed image for visualization
            visualizationImgMsg = Image()
            visualizationImgMsg.header = header
            visualizationImgMsg = self.bridge.cv2_to_imgmsg(
                segmentedImage, "bgr8")
            segmenterData.segmentedImageProbability = pcdProbabilities
            self.publisherSegVis.publish(visualizationImgMsg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


# Run the program
if __name__ == '__main__':
    # Initialization
    rospy.init_node('segmenter', anonymous=False)
    segmenter = Segmenter()
    rospy.spin()
