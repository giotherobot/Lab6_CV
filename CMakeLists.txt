cmake_minimum_required(VERSION 3.2)

project(lab6)

if(DEFINED ENV{OPENCV_INSTALL_DIR})
    find_package(OpenCV 4.3 REQUIRED PATHS $ENV{OPENCV_INSTALL_DIR})
else()
    message("OPENCV_INSTALL_DIR not set, searching in default location(s)")
    find_package(OpenCV 4.3 REQUIRED)
endif(DEFINED ENV{OPENCV_INSTALL_DIR})

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
)

add_executable(
    ${PROJECT_NAME}
    src/main.cpp
)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)



