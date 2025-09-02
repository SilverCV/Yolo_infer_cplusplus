cmake_minimum_required(VERSION 3.31)
project(Yolo)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Yolo main.cpp
        Yolo.cpp
        Yolo.h)

target_link_libraries(Yolo ${OpenCV_LIBRARIES})
