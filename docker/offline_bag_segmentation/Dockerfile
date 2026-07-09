ARG BASE_IMAGE=vsgraphs_ros2
ARG USERNAME=user
ARG ROS_DISTRO=jazzy

FROM ${BASE_IMAGE}
ARG USERNAME=user
ARG ROS_DISTRO=jazzy

USER root
RUN ["bash", "-lc", "apt-get update && apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-rosbag2-storage-mcap && rm -rf /var/lib/apt/lists/*"]
COPY scene_segment_ros/docker/offline_bag_segmentation/offline_bag_segmentation_entrypoint.sh \
  /usr/local/bin/offline_bag_segmentation_entrypoint.sh
RUN ["chmod", "+x", "/usr/local/bin/offline_bag_segmentation_entrypoint.sh"]

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace

ENTRYPOINT ["/usr/local/bin/offline_bag_segmentation_entrypoint.sh"]
CMD ["--help"]
