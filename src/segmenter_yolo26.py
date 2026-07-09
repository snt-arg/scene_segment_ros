#!/usr/bin/env python3

import torch
import rclpy
from rclpy.node import Node
from ultralytics import YOLO
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from output import yoloVisualizer
from output import entropyVisualizer
from modelRunner import yoloSegmenter

from utils.helpers import cleanMemory, monitorParams
from utils.semantic_utils import probabilities2ROSMsg
from ament_index_python import get_package_share_directory
from segmenter_ros.msg import SegmenterDataMsg, VSGraphDataMsg


def allow_trusted_ultralytics_checkpoint_load():
    if getattr(torch.load, "_vsgraphs_trusted_checkpoint_patch", False):
        return

    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch_load_compat._vsgraphs_trusted_checkpoint_patch = True
    torch.load = torch_load_compat


class Segmenter(Node):
    def __init__(self):
        # Variables
        self.pkg_share_directory = get_package_share_directory("segmenter_ros")
        super().__init__(
            "segmenter_ros",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Initial checks
        monitorParams()
        cleanMemory()

        # Configuration parameters
        ground_ids = (
            self.get_parameter("params.output.classes.ground")
            .get_parameter_value()
            .integer_array_value.tolist()
        )

        wall_ids = (
            self.get_parameter("params.output.classes.wall")
            .get_parameter_value()
            .integer_array_value.tolist()
        )

        self.classes = [ground_ids, wall_ids]

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

        self.visualize = (
            self.get_parameter("visualize").get_parameter_value().bool_value
        )

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

        self.imageSize = (
            self.get_parameter("params.image_params.image_size")
            .get_parameter_value()
            .integer_value
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

        # Initial the segmentation module
        allow_trusted_ultralytics_checkpoint_load()
        self.model = YOLO(modelPath, task="semantic")

        # Subscribers (to vS-Graphs)
        self.create_subscription(Image, rawImageTopic, self.segmentation, 10)

        # Publishers (for vS-Graphs)
        self.publisherSeg = self.create_publisher(SegmenterDataMsg, segImageTopic, 10)

        self.publisherSegVis = self.create_publisher(Image, segImageVisTopic, 10)

        # ROS Bridge
        self.bridge = CvBridge()

    def segmentation(self, imageMessage):
        try:
            # Parse the input data
            # key_frame_id = imageMessage.key_frame_id
            # key_frame_image = imageMessage
            inputHeader = imageMessage.header
            # Convert the ROS Image message to a CV2 image
            cvImage = self.bridge.imgmsg_to_cv2(imageMessage, "bgr8")

            # Processing
            filteredSegments, filteredProbs = yoloSegmenter(
                cvImage, self.model, self.classes
            )

            if self.visualize:
                segmented_image = yoloVisualizer(
                    cvImage,
                    filteredSegments,
                )

            segmentedUncImage = entropyVisualizer(
                torch.from_numpy(filteredProbs).permute(2, 0, 1)
            )

            # Convert to ROS message
            pcdProbabilities = probabilities2ROSMsg(
                filteredProbs, imageMessage.header.stamp, imageMessage.header.frame_id
            )

            # Create a header with the current time
            header = Header()
            # now = self.get_clock().now().to_msg()
            header.stamp = inputHeader.stamp
            header.frame_id = inputHeader.frame_id

            # Publish the processed image to vS-Graphs
            segmenterData = SegmenterDataMsg()
            segmenterData.header = header
            # segmenterData.key_frame_id = key_frame_id
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
            self.get_logger().error(f"[Segmenter] CvBridge error: {e}")


def main(args=None):
    # Variables
    node = None
    rclpy.init(args=args)
    #
    try:
        node = Segmenter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Segmenter] Node interrupted by user! Exiting...")
    except Exception as e:
        rclpy.logging.get_logger("YOLO26").error(
            f"[Segmenter] Unhandled exception: {e}"
        )
    finally:
        if node is not None:
            node.destroy_node()


if __name__ == "__main__":
    main()
