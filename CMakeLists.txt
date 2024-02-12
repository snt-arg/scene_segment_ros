cmake_minimum_required(VERSION 3.0.2)
project(segmenter_ros)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  std_msgs
  sensor_msgs
)

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  VSGraphDataMsg.msg
  SegmenterDataMsg.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

# catkin specific configuration ##
catkin_package()

# Build
include_directories(
  ${catkin_INCLUDE_DIRS}
)
