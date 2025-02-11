cmake_minimum_required(VERSION 3.10)
project(dial_mpc)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_definitions(-D_GLIBCXX_PARALLEL)

set(CMAKE_CXX_FLAGS "-O3 -no-pie -march=native -ffast-math -DNDEBUG -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)
find_package(Gnuplot REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem iostreams)

include_directories(${EIGEN3_INCLUDE_DIRS} ${mujoco_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_executable(dial_simulate src/dial-core-simulate.cpp)
target_link_libraries(dial_simulate Eigen3::Eigen mujoco glfw Threads::Threads ${CMAKE_DL_LIBS} ${Boost_LIBRARIES})
target_include_directories(dial_simulate PRIVATE ${PROJECT_SOURCE_DIR}/include)
