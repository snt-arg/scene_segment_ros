#!/usr/bin/env python

import torch
import rospy
from std_msgs.msg import Header
from modelRunner import FCNSegmenter, FCNInit
from cv_bridge import CvBridge, CvBridgeError
from utils.helpers import cleanMemory, monitorParams
from utils.semantic_utils import probabilities2ROSMsg
from output import FCNVisualizer, FCNEntropyVisualizer
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

        # Initial the segmentation module
        self.model, self.cfg = FCNInit(modelName, modelPath, modelConfig)

        # Subscribers (to vS-Graphs)
        rospy.Subscriber(rawImageTopic, VSGraphDataMsg, self.segmentation)

        # Publishers (for vS-Graphs)
        self.publisherSeg = rospy.Publisher(
            segImageTopic, SegmenterDataMsg, queue_size=10)

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
            predictions = FCNSegmenter(cvImage, self.model, self.classes)
            segmentedImage = FCNVisualizer(cvImage, predictions, self.cfg)
            segmentedUncImage = FCNEntropyVisualizer(
                cvImage, predictions, self.cfg)
            prediction_probs = torch.permute(
                predictions["sem_seg"], (1, 2, 0)).to("cpu").numpy()
            pcdProbabilities = probabilities2ROSMsg(prediction_probs,
                                                    imageMessage.header.stamp, imageMessage.header.frame_id)

            # Create a header with the current time
            header = Header()
            header.stamp = rospy.Time.now()

            # Publish the processed image
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

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))


# Run the program
if __name__ == '__main__':
    # Initialization
    rospy.init_node('segmenter', anonymous=False)
    segmenter = Segmenter()
    rospy.spin()
