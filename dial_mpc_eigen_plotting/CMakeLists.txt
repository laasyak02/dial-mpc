cmake_minimum_required(VERSION 3.10)
project(dial_mpc_eigen)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_definitions(-D_GLIBCXX_PARALLEL)

# set(CMAKE_CXX_FLAGS "-O3 -no-pie -march=native -ffast-math -DNDEBUG -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)
find_package(Gnuplot REQUIRED)
find_package(Boost REQUIRED COMPONENTS 
    system 
    filesystem 
    iostreams
)
include_directories(${EIGEN3_INCLUDE_DIRS} ${mujoco_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_executable(test_dial_mpc src/dial-core-test.cpp)
target_link_libraries(test_dial_mpc Eigen3::Eigen mujoco glfw Threads::Threads ${CMAKE_DL_LIBS})
target_include_directories(test_dial_mpc PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(go2_env_test src/unitree-go2-env-test.cpp)
target_link_libraries(go2_env_test Eigen3::Eigen mujoco glfw Threads::Threads ${CMAKE_DL_LIBS})
target_include_directories(go2_env_test PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(dial_simulate src/dial-core-simulate.cpp)
target_link_libraries(dial_simulate Eigen3::Eigen mujoco glfw Threads::Threads ${CMAKE_DL_LIBS} ${Boost_LIBRARIES})
target_include_directories(dial_simulate PRIVATE ${PROJECT_SOURCE_DIR}/include)

# add_executable(test-poly src/test-poly.cpp)
# target_link_libraries(test-poly Eigen3::Eigen)
# target_include_directories(test-poly PRIVATE ${PROJECT_SOURCE_DIR}/include)

# add_executable(main src/main.cpp)
# target_link_libraries(main Eigen3::Eigen mujoco glfw Threads::Threads ${CMAKE_DL_LIBS})
# target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include)