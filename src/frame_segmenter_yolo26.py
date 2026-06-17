#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import os
import rclpy
from rclpy.node import Node
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils.helpers import cleanMemory, monitorParams
from ament_index_python import get_package_share_directory


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
        self.movableClassIds = set(
            self.get_parameter("params.output.classes.movable")
            .get_parameter_value()
            .integer_array_value.tolist()
        )

        self.conf = (
            self.get_parameter("params.model_params.conf")
            .get_parameter_value()
            .double_value
        )

        modelPath = (
            self.get_parameter("params.model_params.model_path")
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

        instanceMaskTopic = (
            self.get_parameter("params.ros_topics.instance_mask_topic")
            .get_parameter_value()
            .string_value
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"[Segmenter] Running YOLO26 on {self.device}")

        # Initial the segmentation module
        if not os.path.isabs(modelPath):
            modelPath = os.path.join(self.pkg_share_directory, modelPath)
        self.model = YOLO(modelPath, task="segment")
        self.model.to(self.device)

        # Subscribers (to vS-Graphs)
        self.create_subscription(Image, rawImageTopic, self.segmentation, 10)

        # Publishers (for vS-Graphs)
        self.publisherInstanceMasks = self.create_publisher(
            Image, instanceMaskTopic, 10
        )

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

            predictions = self.model.predict(
                source=cvImage,
                task="segment",
                imgsz=self.imageSize,
                conf=self.conf,
                device=self.device,
                verbose=False,
            )
            movableMask = self.movable_instance_mask(predictions[0], cvImage.shape[:2])

            instanceMaskMsg = self.bridge.cv2_to_imgmsg(movableMask, "mono8")
            instanceMaskMsg.header = inputHeader
            self.publisherInstanceMasks.publish(instanceMaskMsg)

        except CvBridgeError as e:
            self.get_logger().error(f"[Segmenter] CvBridge error: {e}")

    def movable_instance_mask(self, result, imageShape):
        h, w = imageShape
        outputMask = np.zeros((h, w), dtype=np.uint8)

        resultMasks = getattr(result, "masks", None)
        resultBoxes = getattr(result, "boxes", None)
        if resultMasks is None or resultBoxes is None:
            return outputMask

        masks = getattr(resultMasks, "data", None)
        classes = getattr(resultBoxes, "cls", None)
        if masks is None or classes is None or masks.numel() == 0:
            return outputMask

        masks = masks.detach().to("cpu").numpy()
        classes = classes.detach().to("cpu").numpy().astype(np.int32)

        for mask, classId in zip(masks, classes):
            if int(classId) not in self.movableClassIds:
                continue

            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            outputMask[mask > 0.5] = 255

        return outputMask


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
