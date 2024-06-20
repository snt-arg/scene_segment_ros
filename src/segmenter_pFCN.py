#!/usr/bin/env python

import torch
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from modelRunner import pFCNSegmenter, FCNInit
from cv_bridge import CvBridge, CvBridgeError
from utils.helpers import cleanMemory, monitorParams
from utils.semantic_utils import probabilities2ROSMsg
from output import pFCNVisualizer, pFCNEntropyVisualizer
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
        modelName = params['model_params']['model_name']
        modelPath = params['model_params']['model_path']
        modelConfig = params['model_params']['model_config']
        self.imageSize = params['image_params']['image_size']
        rawImageTopic = params['ros_topics']['raw_image_topic']
        segImageTopic = params['ros_topics']['segmented_image_topic']
        segImageVisTopic = params['ros_topics']['segmented_image_vis']

        # Initial the segmentation module
        self.model, self.cfg = FCNInit(modelName, modelPath, modelConfig)

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
            predictions = pFCNSegmenter(cvImage, self.model, self.classes)
            segmentedImage = pFCNVisualizer(cvImage, predictions, self.cfg)
            segmentedUncImage = pFCNEntropyVisualizer(predictions["sem_seg"])
            predictionProbs = torch.permute(
                predictions["sem_seg"], (1, 2, 0)).to("cpu").numpy()

            # Take only the probabilities for classes needed from params
            predictionProbs = np.take(predictionProbs, self.classes, -1)

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
            segmenterData.segmentedImageUncertainty = self.bridge.cv2_to_imgmsg(
                segmentedUncImage, "bgr8")
            labels = torch.argmax(predictions["sem_seg"], axis=0)
            unique_classes = torch.unique(labels)
            segmenterData.segmentedImageProbability = pcdProbabilities
            self.publisherSeg.publish(segmenterData)

            # Publish the processed image for visualization
            visualizationImgMsg = Image()
            visualizationImgMsg.header = header
            visualizationImgMsg = self.bridge.cv2_to_imgmsg(
                segmentedImage, "bgr8")
            self.publisherSegVis.publish(visualizationImgMsg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


# Run the program
if __name__ == '__main__':
    # Initialization
    rospy.init_node('segmenter', anonymous=False)
    segmenter = Segmenter()
    rospy.spin()
