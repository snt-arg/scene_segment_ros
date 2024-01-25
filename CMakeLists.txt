cmake_minimum_required(VERSION 3.0.2)
project(segmenter_ros)

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  message_generation
  std_msgs
  sensor_msgs
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

# catkin specific configuration ##
catkin_package(
  CATKIN_DEPENDS roscpp message_runtime std_msgs sensor_msgs
)

# Add executables
add_executable(pcd_listener src/pcd_listener.cpp)
target_link_libraries(pcd_listener ${catkin_LIBRARIES})

# Build
include_directories(
  ${catkin_INCLUDE_DIRS}
)
