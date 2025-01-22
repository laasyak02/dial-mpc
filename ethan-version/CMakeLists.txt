cmake_minimum_required(VERSION 3.10)
project(dial_mpc_eigen_corrected)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(dial_mpc src/dial-core3.cpp)