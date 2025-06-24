#!/usr/bin/env python

# import torch
# import rclpy
# import numpy as np
# from std_msgs.msg import Header
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
# from modelRunner import yosoInit, yosoSegmenter
# from utils.helpers import cleanMemory, monitorParams
# from utils.semantic_utils import probabilities2ROSMsg
# from output import yosoVisualizer, entropyVisualizer
# from segmenter_ros.msg import SegmenterDataMsg, VSGraphDataMsg


# from rclpy.node import Node
# from std_msgs.msg import Empty
# from ament_index_python import get_package_share_directory


# class Segmenter(Node):
#     def __init__(self):
#         super().__init__("segmenter")
#         # Initial checks
#         monitorParams()
#         cleanMemory()

#         # self.pkg_share_directory = get_package_share_directory("ros_marker_command_gen")
#         # print(self.pkg_share_directory)

#         # Get parameters
#         print("Loading configuration parameters ...\n")
#         params = rclpy.get_param("~params")
#         self.classes = params["output"]["classes"]
#         self.conf = params["model_params"]["conf"]
#         self.overlap = params["model_params"]["overlap"]
#         modelName = params["model_params"]["model_name"]
#         modelPath = params["model_params"]["model_path"]
#         modelConfig = params["model_params"]["model_config"]
#         self.imageSize = params["image_params"]["image_size"]
#         rawImageTopic = params["ros_topics"]["raw_image_topic"]
#         segImageTopic = params["ros_topics"]["segmented_image_topic"]
#         segImageVisTopic = params["ros_topics"]["segmented_image_vis"]

#         self.visualize = rclpy.get_param("~visualize")

#         # Initial the segmentation module
#         self.model, self.cfg = yosoInit(
#             modelName, modelPath, modelConfig, self.conf, self.overlap
#         )

#         # Subscribers (to vS-Graphs)
#         rclpy.Subscriber(rawImageTopic, VSGraphDataMsg, self.segmentation)

#         # Publishers (for vS-Graphs)
#         self.publisherSeg = rclpy.Publisher(
#             segImageTopic, SegmenterDataMsg, queue_size=10
#         )
#         self.publisherSegVis = rclpy.Publisher(segImageVisTopic, Image, queue_size=10)

#         # ROS Bridge
#         self.bridge = CvBridge()

#     def segmentation(self, imageMessage):
#         try:
#             # Parse the input data
#             keyFrameId = imageMessage.keyFrameId
#             keyFrameImage = imageMessage.keyFrameImage

#             # Convert the ROS Image message to a CV2 image
#             cvImage = self.bridge.imgmsg_to_cv2(keyFrameImage, "bgr8")

#             # Processing
#             filteredSegments, filteredProbs = yosoSegmenter(
#                 cvImage, self.model, self.classes
#             )
#             if self.visualize:
#                 segmentedImage = yosoVisualizer(cvImage, filteredSegments, self.cfg)
#             segmentedUncImage = entropyVisualizer(filteredSegments["sem_seg"])

#             # Convert to ROS message
#             pcdProbabilities = probabilities2ROSMsg(
#                 filteredProbs, imageMessage.header.stamp, imageMessage.header.frame_id
#             )

#             # Create a header with the current time
#             header = Header()
#             header.stamp = rclpy.Time.now()

#             # Publish the processed image to vS-Graphs
#             segmenterData = SegmenterDataMsg()
#             segmenterData.header = header
#             segmenterData.keyFrameId = keyFrameId
#             if self.visualize:
#                 segmenterData.segmentedImage = self.bridge.cv2_to_imgmsg(
#                     segmentedImage, "bgr8"
#                 )
#             segmenterData.segmentedImageUncertainty = self.bridge.cv2_to_imgmsg(
#                 segmentedUncImage, "bgr8"
#             )
#             segmenterData.segmentedImageProbability = pcdProbabilities
#             self.publisherSeg.publish(segmenterData)

#             if self.visualize:
#                 # Publish the processed image for visualization
#                 visualizationImgMsg = Image()
#                 visualizationImgMsg.header = header
#                 visualizationImgMsg = segmenterData.segmentedImage
#                 self.publisherSegVis.publish(visualizationImgMsg)

#         except CvBridgeError as e:
#             # rclpy.logerr("CvBridge Error: {0}".format(e))
#             self.get_logger().warn("Fail of CvBridgeError")


# # Run the program
# # if __name__ == "__main__":
# #     # Initialization
# #     # rclpy.init_node("segmenter", anonymous=False)
# #     rclpy.init("segmenter")
# def main(args=None):
#     # Intialize the ROS2 system (rclpy.init)
#     rclpy.init(args=args)
#     node = Segmenter()
#     try:
#         # Spin function to keep the node alive
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         # Shutdown the node and ROS2 system
#         node.destroy_node()
#         rclpy.shutdown()


# # Run the main function
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3

import torch
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from modelRunner import yosoInit, yosoSegmenter
from utils.helpers import cleanMemory, monitorParams
from utils.semantic_utils import probabilities2ROSMsg
from output import yosoVisualizer, entropyVisualizer
from ament_index_python import get_package_share_directory
from segmenter_ros.msg import SegmenterDataMsg, VSGraphDataMsg


class Segmenter(Node):
    def __init__(self):
        # super().__init__("segmenter")
        self.pkg_share_directory = get_package_share_directory("segmenter_ros")
        super().__init__(
            "segmenter",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        # Initial checks
        monitorParams()
        cleanMemory()

        # Declare and get parameters
        # self.declare_parameter("params", None)
        # self.declare_parameter("visualize", True)
        # # params = self.get_parameter("params").get_parameter_value().string_value
        # # params = self.get_parameter("params").get_parameter_value().value
        # params = self.get_parameter("params").value
        # self.visualize = (
        #     self.get_parameter("visualize").get_parameter_value().bool_value
        # )

        # If params is a YAML string, parse it
        # import yaml

        # if isinstance(params, str):
        #     params = yaml.safe_load(params)
        # elif isinstance(params, dict):
        #     pass  # params is already a dict
        # else:
        #     self.get_logger().error("No 'params' parameter provided!")
        #     params = {}

        # list all parameters

        # self.classes = params.get("output", {}).get("classes", [])
        self.classes = (
            self.get_parameter("params.output.classes")
            .get_parameter_value()
            .integer_array_value.tolist()
        )
        print(f"Classes: {self.classes}")

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

        # self.overlap = params.get("model_params", {}).get("overlap", 0.5)
        # modelName = params.get("model_params", {}).get("model_name", "")
        # modelPath = params.get("model_params", {}).get("model_path", "")
        # modelConfig = params.get("model_params", {}).get("model_config", "")
        # print(f"Loading model: {modelName} from {modelPath} with config {modelConfig}")
        # self.imageSize = params.get("image_params", {}).get("image_size", 640)
        # rawImageTopic = params.get("ros_topics", {}).get(
        #     "raw_image_topic", "/raw_image"
        # )
        # segImageTopic = params.get("ros_topics", {}).get(
        #     "segmented_image_topic", "/segmented_image"
        # )
        # segImageVisTopic = params.get("ros_topics", {}).get(
        #     "segmented_image_vis", "/segmented_image_vis"
        # )

        # Initial the segmentation module
        self.model, self.cfg = yosoInit(
            modelName, modelPath, modelConfig, self.conf, self.overlap
        )

        # Subscribers (to vS-Graphs)
        self.create_subscription(VSGraphDataMsg, rawImageTopic, self.segmentation, 10)

        # Publishers (for vS-Graphs)
        self.publisherSeg = self.create_publisher(SegmenterDataMsg, segImageTopic, 10)
        self.publisherSegVis = self.create_publisher(Image, segImageVisTopic, 10)

        # ROS Bridge
        self.bridge = CvBridge()

    def segmentation(self, imageMessage):
        print("Segmenter: segmentation callback called")
        try:
            # Parse the input data
            key_frame_id = imageMessage.key_frame_id
            key_frame_image = imageMessage.key_frame_image

            # Convert the ROS Image message to a CV2 image
            cvImage = self.bridge.imgmsg_to_cv2(key_frame_image, "bgr8")

            # Processing
            filteredSegments, filteredProbs = yosoSegmenter(
                cvImage, self.model, self.classes
            )
            if self.visualize:
                segmented_image = yosoVisualizer(cvImage, filteredSegments, self.cfg)
            segmentedUncImage = entropyVisualizer(filteredSegments["sem_seg"])

            # Convert to ROS message
            pcdProbabilities = probabilities2ROSMsg(
                filteredProbs, imageMessage.header.stamp, imageMessage.header.frame_id
            )

            # Create a header with the current time
            header = Header()
            now = self.get_clock().now().to_msg()
            header.stamp = now
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
