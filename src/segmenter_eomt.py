#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from modelRunner import eomtInit, eomtSegmenter
from utils.helpers import cleanMemory, monitorParams
from output import eomtVisualizer, entropyVisualizer
from utils.semantic_utils import probabilities2ROSMsg
from ament_index_python import get_package_share_directory
from segmenter_ros.msg import SegmenterDataMsg, VSGraphDataMsg


class Segmenter(Node):
    def __init__(self):
        self.pkg_share_directory = get_package_share_directory("segmenter_ros")
        super().__init__(
            "segmenter",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        monitorParams()
        cleanMemory()

        ground_ids = self.get_parameter(
            "params.output.classes.ground"
        ).get_parameter_value().integer_array_value.tolist()
        wall_ids = self.get_parameter(
            "params.output.classes.wall"
        ).get_parameter_value().integer_array_value.tolist()
        door_ids = self.get_parameter(
            "params.output.classes.door"
        ).get_parameter_value().integer_array_value.tolist()
        self.classes = [ground_ids, wall_ids, door_ids]

        self.conf = (
            self.get_parameter("params.model_params.conf")
            .get_parameter_value()
            .double_value
        )
        self.overlap = (
            self.get_parameter("params.model_params.overlap")
            .get_parameter_value()
            .double_value
        )
        self.visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        modelName = (
            self.get_parameter("params.model_params.model_name")
            .get_parameter_value()
            .string_value
        )
        modelPath = (
            self.get_parameter("params.model_params.model_path")
            .get_parameter_value()
            .string_value
        )
        modelConfig = (
            self.get_parameter("params.model_params.model_config")
            .get_parameter_value()
            .string_value
        )
        rawImageTopic = (
            self.get_parameter("params.ros_topics.raw_image_topic")
            .get_parameter_value()
            .string_value
        )
        segImageTopic = (
            self.get_parameter("params.ros_topics.segmented_image_topic")
            .get_parameter_value()
            .string_value
        )
        segImageVisTopic = (
            self.get_parameter("params.ros_topics.segmented_image_vis")
            .get_parameter_value()
            .string_value
        )

        self.model, self.processor, self.cfg = eomtInit(
            modelName, modelPath, modelConfig, self.conf, self.overlap
        )

        self.create_subscription(VSGraphDataMsg, rawImageTopic, self.segmentation, 10)
        self.publisherSeg = self.create_publisher(SegmenterDataMsg, segImageTopic, 10)
        self.publisherSegVis = self.create_publisher(Image, segImageVisTopic, 10)
        self.bridge = CvBridge()

    def segmentation(self, imageMessage):
        try:
            key_frame_id = imageMessage.key_frame_id
            key_frame_image = imageMessage.key_frame_image
            cvImage = self.bridge.imgmsg_to_cv2(key_frame_image, "bgr8")

            filteredSegments, filteredProbs = eomtSegmenter(
                cvImage, self.model, self.processor, self.cfg, self.classes
            )
            if self.visualize:
                segmented_image = eomtVisualizer(cvImage, filteredSegments, self.cfg)
            segmentedUncImage = entropyVisualizer(filteredSegments["sem_seg"])

            pcdProbabilities = probabilities2ROSMsg(
                filteredProbs, imageMessage.header.stamp, imageMessage.header.frame_id
            )

            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = imageMessage.header.frame_id

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
                visualizationImgMsg = segmenterData.segmented_image
                visualizationImgMsg.header = header
                self.publisherSegVis.publish(visualizationImgMsg)

        except CvBridgeError as e:
            self.get_logger().error(f"[Segmenter] CvBridge error: {e}")


def main(args=None):
    node = None
    rclpy.init(args=args)
    try:
        node = Segmenter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("[Segmenter] Node interrupted by user! Exiting...")
    except Exception as e:
        rclpy.logging.get_logger().error(f"[Segmenter] Unhandled exception: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
