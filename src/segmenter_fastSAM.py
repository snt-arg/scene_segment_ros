#!/usr/bin/env python

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from output import fastSamVisualizer
from cv_bridge import CvBridge, CvBridgeError
from utils.helpers import cleanMemory, monitorParams
from modelRunner import fastSamInit, fastSamSegmenter
from segmenter_ros.msg import SegmenterDataMsg, VSGraphDataMsg


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
        self.counters = params['model_params']['contour']
        self.boxPrompt = params['model_params']['box_prompt']
        self.imageSize = params['image_params']['image_size']
        rawImageTopic = params['ros_topics']['raw_image_topic']
        self.pointLabel = params['model_params']['point_label']
        self.pointPrompt = params['model_params']['point_prompt']
        segImageTopic = params['ros_topics']['segmented_image_topic']
        segImageVisTopic = params['ros_topics']['segmented_image_vis']

        # Initial the segmentation module
        self.fsam = fastSamInit(modelName, modelPath)

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
            masks = fastSamSegmenter(
                cvImage, self.fsam, self.imageSize, self.conf, self.iou)
            segmentedImage = fastSamVisualizer(
                masks, self.pointPrompt, self.boxPrompt, self.pointLabel, self.counters)

            # Create a header with the current time
            header = Header()
            header.stamp = rospy.Time.now()

            # Publish the processed image to vS-Graphs
            segmenterData = SegmenterDataMsg()
            segmenterData.header = header
            segmenterData.keyFrameId = keyFrameId
            segmenterData.segmentedImage = self.bridge.cv2_to_imgmsg(
                segmentedImage, "bgr8")
            # [TODO] Add the segmentation uncertainty
            # segmenterData.segmentedImageUncertainty =
            # [TODO] Add the segmentation probability
            # segmenterData.segmentedImageProbability =
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
