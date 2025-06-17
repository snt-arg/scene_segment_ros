cmake_minimum_required(VERSION 3.5)
project(segmenter_ros)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Message generation
set(msg_files
  "msg/VSGraphDataMsg.msg"
  "msg/SegmenterDataMsg.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs sensor_msgs
)

# ament export
ament_export_dependencies(rosidl_default_runtime)

# Install Python scripts
install(PROGRAMS
  src/segmenter_pFCN.py
  src/segmenter_yoso.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install other needed files
install(DIRECTORY
  config launch src
  DESTINATION share/${PROJECT_NAME}
)

ament_package()