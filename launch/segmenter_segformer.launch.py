import os

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    node_name_arg = DeclareLaunchArgument(
        "node_name",
        default_value="segmenter_ros"
    )

    visualize_arg = DeclareLaunchArgument(
        "visualize",
        default_value="true"
    )

    config_file_path = os.path.join(
        get_package_share_directory("segmenter_ros"),
        "config",
        "cfg_segformer.yaml"
    )

    return LaunchDescription([
        node_name_arg,
        visualize_arg,
        Node(
            package="segmenter_ros",
            executable="segmenter_segformer.py",
            name=LaunchConfiguration("node_name"),
            output="screen",
            parameters=[
                config_file_path,
                {
                    "visualize": LaunchConfiguration("visualize")
                }
            ]
        )
    ])
