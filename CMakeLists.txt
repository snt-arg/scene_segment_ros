cmake_minimum_required(VERSION 3.0.2)
project(segmenter_ros)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  std_msgs
  sensor_msgs
)

add_message_files(
  FILES 
  Pixel.msg
  Mask.msg
  SegResult.msg
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
