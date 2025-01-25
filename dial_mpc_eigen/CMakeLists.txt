cmake_minimum_required(VERSION 3.10)
project(dial_mpc_eigen)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_definitions(-D_GLIBCXX_PARALLEL)

set(CMAKE_CXX_FLAGS "-O3 -no-pie -march=native -ffast-math -DNDEBUG -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS} ${mujoco_INCLUDE_DIRS})

add_executable(test_dial_mpc src/dial-core-test.cpp)
target_link_libraries(test_dial_mpc mujoco glfw3 Threads::Threads ${CMAKE_DL_LIBS})
target_include_directories(test_dial_mpc PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(go2_env_test src/unitree-go2-env-test.cpp)
target_link_libraries(go2_env_test mujoco glfw Threads::Threads ${CMAKE_DL_LIBS})
target_include_directories(go2_env_test PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(dial_simulate src/dial-core-simulate.cpp)
target_link_libraries(dial_simulate mujoco glfw Threads::Threads ${CMAKE_DL_LIBS})
target_include_directories(dial_simulate PRIVATE ${PROJECT_SOURCE_DIR}/include)