cmake_minimum_required(VERSION 3.10)
project(dial_mpc_eigen)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS} ${mujoco_INCLUDE_DIRS})

add_executable(dial_mpc src/dial-core3.cpp)
target_link_libraries(dial_mpc mujoco glfw3 Threads::Threads ${CMAKE_DL_LIBS})

add_executable(go2_env src/UnitreeGo2Env.cpp)
target_link_libraries(go2_env mujoco glfw3 Threads::Threads ${CMAKE_DL_LIBS})