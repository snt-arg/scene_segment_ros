#!/usr/bin/env python3

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from utils.helpers import cleanMemory, monitorParams
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

        self.bridge = CvBridge()
        self.device = "cpu"

        # -----------------------------
        # Parameters
        # -----------------------------

        self.conf = (
            self.get_parameter("params.model_params.conf")
            .get_parameter_value()
            .double_value
        )

        self.visualize = (
            self.get_parameter("visualize")
            .get_parameter_value()
            .bool_value
        )

        self.model_name = (
            self.get_parameter("params.model_params.model_name")
            .get_parameter_value()
            .string_value
        )

        self.image_size = (
            self.get_parameter("params.image_params.image_size")
            .get_parameter_value()
            .integer_value
        )

        raw_image_topic = (
            self.get_parameter("params.ros_topics.raw_image_topic")
            .get_parameter_value()
            .string_value
        )

        seg_image_topic = (
            self.get_parameter("params.ros_topics.segmented_image_topic")
            .get_parameter_value()
            .string_value
        )

        seg_image_vis_topic = (
            self.get_parameter("params.ros_topics.segmented_image_vis")
            .get_parameter_value()
            .string_value
        )

        # These are kept for compatibility with the old config style.
        # The actual ADE20K class mapping is found from model.config.id2label.
        ground_ids = self.get_parameter(
            "params.output.classes.ground"
        ).get_parameter_value().integer_array_value.tolist()

        wall_ids = self.get_parameter(
            "params.output.classes.wall"
        ).get_parameter_value().integer_array_value.tolist()

        self.classes = [ground_ids, wall_ids]

        # -----------------------------
        # Load SegFormer
        # -----------------------------

        self.get_logger().info(
            f"[Segmenter] Loading SegFormer model: {self.model_name}")

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

        self.ground_ade_ids = self.find_ade_class_ids(["floor", "ground"])
        self.wall_ade_ids = self.find_ade_class_ids(["wall"])

        self.get_logger().info(
            f"[Segmenter] Ground ADE ids: {self.ground_ade_ids}")
        self.get_logger().info(
            f"[Segmenter] Wall ADE ids: {self.wall_ade_ids}")

        if not self.ground_ade_ids:
            self.get_logger().warn(
                "[Segmenter] No ground/floor ADE classes found.")

        if not self.wall_ade_ids:
            self.get_logger().warn("[Segmenter] No wall ADE classes found.")

        # -----------------------------
        # ROS IO
        # -----------------------------

        self.get_logger().info(f"raw_image_topic:=     {raw_image_topic}")
        self.get_logger().info(f"seg_image_topic:=     {seg_image_topic}")
        self.get_logger().info(f"seg_image_vis_topic:= {seg_image_vis_topic}")

        self.create_subscription(
            VSGraphDataMsg,
            raw_image_topic,
            self.segmentation,
            10
        )

        self.publisherSeg = self.create_publisher(
            SegmenterDataMsg,
            seg_image_topic,
            10
        )

        self.publisherSegVis = self.create_publisher(
            Image,
            seg_image_vis_topic,
            10
        )

        self.get_logger().info("[Segmenter] SegFormer segmenter ready.")

    def find_ade_class_ids(self, keywords):
        """
        Find ADE20K class IDs from model.config.id2label by keyword.
        Example labels include:
          'wall'
          'floor, flooring'
        """
        ids = []

        for idx, label in self.id2label.items():
            label_lower = label.lower()

            for keyword in keywords:
                if keyword.lower() in label_lower:
                    ids.append(int(idx))
                    break

        return sorted(list(set(ids)))

    def run_segformer(self, cv_image_bgr):
        """
        Runs SegFormer and returns probability maps:

            filtered_probs[0] = ground/floor probability
            filtered_probs[1] = wall probability

        Shape:
            2 x H x W

        Type:
            float32
        """
        original_h, original_w = cv_image_bgr.shape[:2]

        cv_image_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)

        # Optional resize for speed.
        # SegFormer processor will also resize internally, but this gives you
        # predictable CPU cost.
        if self.image_size > 0:
            scale = self.image_size / max(original_h, original_w)
            if scale < 1.0:
                resized_w = int(original_w * scale)
                resized_h = int(original_h * scale)
                cv_image_rgb_small = cv2.resize(
                    cv_image_rgb,
                    (resized_w, resized_h),
                    interpolation=cv2.INTER_AREA
                )
            else:
                cv_image_rgb_small = cv_image_rgb
        else:
            cv_image_rgb_small = cv_image_rgb

        inputs = self.processor(
            images=cv_image_rgb_small,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)[0]  # C x h x w
        probs_np = probs.cpu().numpy().astype(np.float32)

        # Resize each class probability map back to original image size
        class_probs = []

        for c in range(probs_np.shape[0]):
            class_prob = cv2.resize(
                probs_np[c],
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR
            )
            class_probs.append(class_prob)

        class_probs = np.stack(class_probs, axis=0)

        ground_prob = np.zeros((original_h, original_w), dtype=np.float32)
        wall_prob = np.zeros((original_h, original_w), dtype=np.float32)

        for class_id in self.ground_ade_ids:
            if class_id < class_probs.shape[0]:
                ground_prob = np.maximum(ground_prob, class_probs[class_id])

        for class_id in self.wall_ade_ids:
            if class_id < class_probs.shape[0]:
                wall_prob = np.maximum(wall_prob, class_probs[class_id])

        # Optional confidence threshold.
        # This keeps low-confidence regions from being interpreted as useful geometry.
        if self.conf > 0.0:
            ground_prob = np.where(ground_prob >= self.conf, ground_prob, 0.0)
            wall_prob = np.where(wall_prob >= self.conf, wall_prob, 0.0)

        filtered_probs = np.stack(
            [ground_prob, wall_prob],
            axis=0
        ).astype(np.float32)

        return filtered_probs

    def make_visualization(self, cv_image_bgr, filtered_probs):
        """
        Debug visualization:
          green = ground/floor
          red = wall
        """
        vis = cv_image_bgr.copy()

        ground_prob = filtered_probs[0]
        wall_prob = filtered_probs[1]

        ground_mask = ground_prob > self.conf
        wall_mask = wall_prob > self.conf

        overlay = np.zeros_like(vis)

        # BGR colors
        overlay[ground_mask] = (0, 220, 0)
        overlay[wall_mask] = (0, 0, 220)

        vis = cv2.addWeighted(vis, 0.70, overlay, 0.30, 0)

        cv2.putText(
            vis,
            "green=ground/floor, red=wall",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return vis

    def make_uncertainty_image(self, filtered_probs):
        """
        Simple uncertainty visualization.

        High uncertainty = no strong ground/wall probability.
        """
        max_prob = np.max(filtered_probs, axis=0)
        uncertainty = 1.0 - max_prob

        uncertainty_u8 = np.clip(
            uncertainty * 255.0,
            0,
            255
        ).astype(np.uint8)

        uncertainty_bgr = cv2.applyColorMap(
            uncertainty_u8,
            cv2.COLORMAP_INFERNO
        )

        return uncertainty_bgr

    def segmentation(self, imageMessage):
        try:
            key_frame_id = imageMessage.key_frame_id
            key_frame_image = imageMessage.key_frame_image

            cv_image = self.bridge.imgmsg_to_cv2(key_frame_image, "bgr8")

            filtered_probs = self.run_segformer(cv_image)

            if self.visualize:
                segmented_image = self.make_visualization(
                    cv_image,
                    filtered_probs
                )

            segmented_unc_image = self.make_uncertainty_image(filtered_probs)

            pcd_probabilities = probabilities2ROSMsg(
                filtered_probs,
                imageMessage.header.stamp,
                imageMessage.header.frame_id
            )

            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = imageMessage.header.frame_id

            segmenter_data = SegmenterDataMsg()
            segmenter_data.header = header
            segmenter_data.key_frame_id = key_frame_id

            if self.visualize:
                segmenter_data.segmented_image = self.bridge.cv2_to_imgmsg(
                    segmented_image,
                    "bgr8"
                )

            segmenter_data.segmented_image_uncertainty = self.bridge.cv2_to_imgmsg(
                segmented_unc_image,
                "bgr8"
            )

            segmenter_data.segmented_image_probability = pcd_probabilities

            self.publisherSeg.publish(segmenter_data)

            if self.visualize:
                visualization_img_msg = segmenter_data.segmented_image
                visualization_img_msg.header = header
                self.publisherSegVis.publish(visualization_img_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"[Segmenter] CvBridge error: {e}")

        except Exception as e:
            self.get_logger().error(
                f"[Segmenter] Error during segmentation: {e}")


def main(args=None):
    node = None
    rclpy.init(args=args)

    try:
        node = Segmenter()
        rclpy.spin(node)

    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info(
                "[Segmenter] Node interrupted by user. Exiting.")

    except Exception as e:
        rclpy.logging.get_logger("segmenter_segformer").error(
            f"[Segmenter] Unhandled exception: {e}"
        )

    finally:
        if node is not None:
            node.destroy_node()


if __name__ == "__main__":
    main()
