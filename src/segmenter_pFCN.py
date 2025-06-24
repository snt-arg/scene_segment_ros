#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from modelRunner import pFCNSegmenter, pFCNInit
from utils.helpers import cleanMemory, monitorParams
from utils.semantic_utils import probabilities2ROSMsg
from output import pFCNVisualizer, entropyVisualizer
from segmenter_ros.msg import SegmenterDataMsg, VSGraphDataMsg


class Segmenter(Node):
    def __init__(self):
        super().__init__("segmenter_pFCN")
        # Initial checks
        monitorParams()
        cleanMemory()

        # Declare and get parameters
        self.declare_parameter("params", None)
        self.declare_parameter("visualize", True)
        params = self.get_parameter("params").get_parameter_value().string_value
        self.visualize = (
            self.get_parameter("visualize").get_parameter_value().bool_value
        )

        import yaml

        if isinstance(params, str):
            params = yaml.safe_load(params)
        else:
            self.get_logger().error("No 'params' parameter provided!")
            params = {}

        self.classes = params.get("output", {}).get("classes", [])
        self.conf = params.get("model_params", {}).get("conf", 0.5)
        modelName = params.get("model_params", {}).get("model_name", "")
        modelPath = params.get("model_params", {}).get("model_path", "")
        modelConfig = params.get("model_params", {}).get("model_config", "")
        self.imageSize = params.get("image_params", {}).get("image_size", 640)
        rawImageTopic = params.get("ros_topics", {}).get(
            "raw_image_topic", "/raw_image"
        )
        segImageTopic = params.get("ros_topics", {}).get(
            "segmented_image_topic", "/segmented_image"
        )
        segImageVisTopic = params.get("ros_topics", {}).get(
            "segmented_image_vis", "/segmented_image_vis"
        )

        # Initialize the segmentation module
        self.model, self.cfg = pFCNInit(modelName, modelPath, modelConfig)

        # Subscribers (to vS-Graphs)
        self.create_subscription(VSGraphDataMsg, rawImageTopic, self.segmentation, 10)

        # Publishers (for vS-Graphs)
        self.publisherSeg = self.create_publisher(SegmenterDataMsg, segImageTopic, 10)
        self.publisherSegVis = self.create_publisher(Image, segImageVisTopic, 10)

        # ROS Bridge
        self.bridge = CvBridge()

    def segmentation(self, imageMessage):
        try:
            # Parse the input data
            key_frame_id = imageMessage.key_frame_id
            key_frame_image = imageMessage.key_frame_image

            # Convert the ROS Image message to a CV2 image
            cvImage = self.bridge.imgmsg_to_cv2(key_frame_image, "bgr8")

            # Processing
            filteredSegments, filteredProbs = pFCNSegmenter(
                cvImage, self.model, self.classes
            )
            if self.visualize:
                segmented_image = pFCNVisualizer(cvImage, filteredSegments, self.cfg)
            segmentedUncImage = entropyVisualizer(filteredSegments["sem_seg"])

            # Convert to ROS message
            pcdProbabilities = probabilities2ROSMsg(
                filteredProbs, imageMessage.header.stamp, imageMessage.header.frame_id
            )

            # Create a header with the current time
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = imageMessage.header.frame_id

            # Publish the processed image to vS-Graphs
            segmenterData = SegmenterDataMsg()
            segmenterData.header = header
            segmenterData.key_frame_id = key_frame_id
            if self.visualize:
                segmenterData.segmented_image = self.bridge.cv2_to_imgmsg(
                    segmented_image, "bgr8"
                )
            segmenterData.segmented_image_uncertainty = self.bridge.cv2_to_imgmsg(
                segmentedUncImage, "bgr8"
            )
            segmenterData.segmented_image_probability = pcdProbabilities
            self.publisherSeg.publish(segmenterData)

            if self.visualize:
                # Publish the processed image for visualization
                visualizationImgMsg = segmenterData.segmented_image
                visualizationImgMsg.header = header
                self.publisherSegVis.publish(visualizationImgMsg)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")


def main(args=None):
    rclpy.init(args=args)
    segmenter = Segmenter()
    rclpy.spin(segmenter)
    segmenter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
