#!/usr/bin/env python3

import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.serialization import deserialize_message, serialize_message
from rosbag2_py import (
    ConverterOptions,
    SequentialReader,
    SequentialWriter,
    StorageOptions,
    TopicMetadata,
)
from sensor_msgs.msg import Image
from ultralytics import YOLO

from segmenter_ros.msg import InstanceMask, InstanceMaskArray


def allow_trusted_ultralytics_checkpoint_load():
    if getattr(torch.load, "_vsgraphs_trusted_checkpoint_patch", False):
        return

    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch_load_compat._vsgraphs_trusted_checkpoint_patch = True
    torch.load = torch_load_compat


class OfflineSegmenter:
    def __init__(self, model_path, image_size, confidence, movable_class_ids):
        self.pkg_share_directory = get_package_share_directory("segmenter_ros")
        self.imageSize = image_size
        self.conf = confidence
        self.movableClassIds = set(movable_class_ids)
        self.bridge = CvBridge()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if not os.path.isabs(model_path):
            model_path = os.path.join(self.pkg_share_directory, model_path)

        allow_trusted_ultralytics_checkpoint_load()
        self.model = YOLO(model_path, task="segment")
        self.model.to(self.device)

    def segmentation(self, imageMessage):
        try:
            inputHeader = imageMessage.header
            cvImage = self.bridge.imgmsg_to_cv2(imageMessage, "bgr8")

            predictions = self.model.predict(
                source=cvImage,
                task="segment",
                imgsz=self.imageSize,
                conf=self.conf,
                device=self.device,
                verbose=False,
            )
            return self.movable_instance_masks(
                predictions[0], cvImage.shape[:2], inputHeader
            )
        except CvBridgeError as e:
            raise RuntimeError(f"CvBridge error while segmenting bag frame: {e}") from e

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


def build_topic_metadata(topic_name, topic_type, offered_qos_profiles=None, topic_id=0):
    if offered_qos_profiles is None:
        offered_qos_profiles = []
    return TopicMetadata(
        topic_id,
        topic_name,
        topic_type,
        "cdr",
        offered_qos_profiles,
    )


def prepare_output_bag(output_bag_path, overwrite):
    if os.path.exists(output_bag_path):
        if not overwrite:
            raise FileExistsError(
                f"Output bag '{output_bag_path}' already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_bag_path)


def augment_bag(args):
    segmenter = OfflineSegmenter(
        model_path=args.model_path,
        image_size=args.image_size,
        confidence=args.confidence,
        movable_class_ids=args.movable_class_ids,
    )

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.input_bag, storage_id="mcap"),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    output_bag_path = args.output_bag
    if output_bag_path is None:
        output_bag_path = f"{args.input_bag.rstrip(os.sep)}_augmented"

    prepare_output_bag(output_bag_path, args.overwrite)

    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=output_bag_path, storage_id="mcap"),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topics = reader.get_all_topics_and_types()
    registered_topics = set()
    for topic in topics:
        writer.create_topic(
            build_topic_metadata(
                topic.name,
                topic.type,
                getattr(topic, "offered_qos_profiles", ""),
            )
        )
        registered_topics.add(topic.name)

    instance_mask_topic = args.instance_mask_topic
    instance_masks_array_topic = args.instance_masks_array_topic
    if not instance_masks_array_topic:
        instance_masks_array_topic = f"{instance_mask_topic}_array"

    if instance_mask_topic not in registered_topics:
        writer.create_topic(
            build_topic_metadata(instance_mask_topic, "sensor_msgs/msg/Image")
        )
    if instance_masks_array_topic not in registered_topics:
        writer.create_topic(
            build_topic_metadata(
                instance_masks_array_topic, "segmenter_ros/msg/InstanceMaskArray"
            )
        )

    processed_frames = 0
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        writer.write(topic_name, data, timestamp)

        if topic_name != args.raw_image_topic:
            continue

        imageMessage = deserialize_message(data, Image)
        movableMask, instanceMasks = segmenter.segmentation(imageMessage)

        instanceMaskMsg = segmenter.bridge.cv2_to_imgmsg(movableMask, "mono8")
        instanceMaskMsg.header = imageMessage.header

        writer.write(instance_mask_topic, serialize_message(instanceMaskMsg), timestamp)
        writer.write(
            instance_masks_array_topic,
            serialize_message(instanceMasks),
            timestamp,
        )
        processed_frames += 1

    print(
        f"Augmented '{args.input_bag}' into '{output_bag_path}' with {processed_frames} segmented frames."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment an MCAP ROS bag with YOLO26 instance mask topics."
    )
    parser.add_argument(
        "input_bag", help="Input rosbag2 directory containing MCAP data."
    )
    parser.add_argument(
        "output_bag",
        nargs="?",
        default=None,
        help="Output rosbag2 directory. Defaults to <input_bag>_augmented.",
    )
    parser.add_argument(
        "--raw-image-topic",
        default="/camera/realsense/color/image_raw",
        help="Topic to segment from the input bag.",
    )
    parser.add_argument(
        "--instance-mask-topic",
        default="/camera/color/image_instance_masks",
        help="Output topic for the full-instance binary mask image.",
    )
    parser.add_argument(
        "--instance-masks-array-topic",
        default="",
        help="Output topic for the InstanceMaskArray message. Defaults to <instance-mask-topic>_array.",
    )
    parser.add_argument(
        "--model-path",
        default="include/yolo26x-seg.pt",
        help="YOLO26 segmentation model path. Relative paths are resolved inside the package share directory.",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--movable-class-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        help="Class ids to keep as movable instances.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output bag if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    augment_bag(args)


if __name__ == "__main__":
    main()