cmake_minimum_required(VERSION 3.5)
project(segmenter_ros)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)

install(PROGRAMS
  src/segmenter_pFCN.py
  src/segmenter_yoso.py
  src/segmenter_yolo26.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install other needed files
install(DIRECTORY
  config launch src
  DESTINATION share/${PROJECT_NAME}
)

ament_package()