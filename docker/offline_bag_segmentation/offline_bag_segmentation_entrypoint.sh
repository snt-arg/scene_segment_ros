#!/usr/bin/env bash
set -eo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
WORKSPACE="${WORKSPACE:-$HOME/workspace}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"

if [ -f "${WORKSPACE}/install/setup.bash" ]; then
  source "${WORKSPACE}/install/setup.bash"
fi

if ! ros2 pkg executables segmenter_ros 2>/dev/null | grep -q bag_segmenter_yolo26.py; then
  echo "[offline_bag_segmentation] bag segmenter not built; building segmenter_ros..."
  cd "${WORKSPACE}"
  colcon build \
    --symlink-install \
    --packages-select segmenter_ros \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
  source "${WORKSPACE}/install/setup.bash"
fi

set -u

exec ros2 run segmenter_ros bag_segmenter_yolo26.py "$@"
