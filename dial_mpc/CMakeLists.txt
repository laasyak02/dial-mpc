cmake_minimum_required(VERSION 3.8)
project(dial_mpc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(mujoco REQUIRED)

if(NOT MUJOCO_DIR)
    set(MUJOCO_DIR "/usr/local/") # or wherever your MuJoCo is installed
endif()

add_executable(control_sequence src/control_sequence.cpp)
add_executable(control_sequence_visualizer src/control_sequence_visualizer.cpp)
add_executable(legged_robot_ik src/legged_robot_ik.cpp)

target_include_directories(control_sequence PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${MUJOCO_DIR}/include)

target_include_directories(control_sequence_visualizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_include_directories(legged_robot_ik PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${MUJOCO_DIR}/include)

target_compile_features(control_sequence PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(control_sequence_visualizer PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(legged_robot_ik PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(control_sequence
  rclcpp
  std_msgs
  geometry_msgs
  mujoco
)

ament_target_dependencies(control_sequence_visualizer
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
)
ament_target_dependencies(legged_robot_ik
  rclcpp
  mujoco
)

link_directories(${MUJOCO_DIR}/lib)

target_link_libraries(control_sequence 
  mujoco
  ${EIGEN3_LIBRARIES}
)

target_link_libraries(legged_robot_ik 
  mujoco
)

install(TARGETS control_sequence control_sequence_visualizer legged_robot_ik
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
