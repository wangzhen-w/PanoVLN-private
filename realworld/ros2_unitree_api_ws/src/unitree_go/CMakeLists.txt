cmake_minimum_required(VERSION 3.5)
project(unitree_go)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TimeSpec.msg"
  "msg/IMUState.msg"
  "msg/PathPoint.msg"
  "msg/SportModeState.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
