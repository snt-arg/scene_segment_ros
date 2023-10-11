cmake_minimum_required(VERSION 3.0.2)
project(segmenter)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
)

# catkin specific configuration ##
catkin_package()

# Build
include_directories(
  ${catkin_INCLUDE_DIRS}
)