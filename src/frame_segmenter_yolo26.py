#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import os
import rclpy
from time import perf_counter
from rclpy.node import Node
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from segmenter_ros.msg import InstanceMask, InstanceMaskArray

from utils.helpers import cleanMemory, monitorParams
from ament_index_python import get_package_share_directory


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
        instanceMasksArrayTopic = (
            self.get_parameter("params.ros_topics.instance_masks_array_topic")
            .get_parameter_value()
            .string_value
        )
        if not instanceMasksArrayTopic:
            instanceMasksArrayTopic = f"{instanceMaskTopic}_array"

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"[Segmenter] Running YOLO26 on {self.device}")

        # Initial the segmentation module
        if not os.path.isabs(modelPath):
            modelPath = os.path.join(self.pkg_share_directory, modelPath)
        allow_trusted_ultralytics_checkpoint_load()
        self.model = YOLO(modelPath, task="segment")
        self.model.to(self.device)

        # Subscribers (to vS-Graphs)
        self.create_subscription(Image, rawImageTopic, self.segmentation, 10)

        # Publishers (for vS-Graphs)
        self.publisherInstanceMasks = self.create_publisher(
            Image, instanceMaskTopic, 10
        )
        self.publisherInstanceMasksArray = self.create_publisher(
            InstanceMaskArray, instanceMasksArrayTopic, 10
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

            segmentationStart = perf_counter()
            predictions = self.model.predict(
                source=cvImage,
                task="segment",
                imgsz=self.imageSize,
                conf=self.conf,
                device=self.device,
                verbose=False,
            )
            self.get_logger().info(
                f"[Segmenter] Segmentation took {(perf_counter() - segmentationStart) * 1000:.2f} ms"
            )
            movableMask, instanceMasks = self.movable_instance_masks(
                predictions[0], cvImage.shape[:2], inputHeader
            )

            instanceMaskMsg = self.bridge.cv2_to_imgmsg(movableMask, "mono8")
            instanceMaskMsg.header = inputHeader
            self.publisherInstanceMasks.publish(instanceMaskMsg)
            self.publisherInstanceMasksArray.publish(instanceMasks)

        except CvBridgeError as e:
            self.get_logger().error(f"[Segmenter] CvBridge error: {e}")

    def movable_instance_masks(self, result, imageShape, header):
        h, w = imageShape
        outputMask = np.zeros((h, w), dtype=np.uint8)
        outputMasks = InstanceMaskArray()
        outputMasks.header = header

        resultMasks = getattr(result, "masks", None)
        resultBoxes = getattr(result, "boxes", None)
        if resultMasks is None or resultBoxes is None:
            return outputMask, outputMasks

        masks = getattr(resultMasks, "data", None)
        classes = getattr(resultBoxes, "cls", None)
        confidences = getattr(resultBoxes, "conf", None)
        boxes = getattr(resultBoxes, "xyxy", None)
        if masks is None or classes is None or masks.numel() == 0:
            return outputMask, outputMasks

        masks = masks.detach().to("cpu").numpy()
        classes = classes.detach().to("cpu").numpy().astype(np.int32)
        if confidences is not None:
            confidences = confidences.detach().to("cpu").numpy().astype(np.float32)
        if boxes is not None:
            boxes = boxes.detach().to("cpu").numpy().astype(np.float32)

        for detectionIndex, (mask, classId) in enumerate(zip(masks, classes)):
            if int(classId) not in self.movableClassIds:
                continue

            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            binaryMask = np.zeros((h, w), dtype=np.uint8)
            binaryMask[mask > 0.5] = 255
            outputMask[binaryMask > 0] = 255

            instanceMask = InstanceMask()
            instanceMask.header = header
            instanceMask.instance_id = int(detectionIndex)
            instanceMask.class_id = int(classId)
            if confidences is not None and detectionIndex < len(confidences):
                instanceMask.score = float(confidences[detectionIndex])
            if boxes is not None and detectionIndex < len(boxes):
                x1, y1, x2, y2 = boxes[detectionIndex]
                x1 = int(np.clip(np.floor(x1), 0, w))
                y1 = int(np.clip(np.floor(y1), 0, h))
                x2 = int(np.clip(np.ceil(x2), 0, w))
                y2 = int(np.clip(np.ceil(y2), 0, h))
                instanceMask.roi.x_offset = x1
                instanceMask.roi.y_offset = y1
                instanceMask.roi.width = max(0, x2 - x1)
                instanceMask.roi.height = max(0, y2 - y1)
                instanceMask.roi.do_rectify = False

            roi = instanceMask.roi
            if roi.width == 0 or roi.height == 0:
                continue

            maskCrop = binaryMask[
                roi.y_offset : roi.y_offset + roi.height,
                roi.x_offset : roi.x_offset + roi.width,
            ]
            maskMsg = self.bridge.cv2_to_imgmsg(maskCrop, "mono8")
            maskMsg.header = header
            instanceMask.mask = maskMsg
            outputMasks.instances.append(instanceMask)

        return outputMask, outputMasks


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
