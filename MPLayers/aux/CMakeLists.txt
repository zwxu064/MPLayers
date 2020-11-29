project(AUX)
cmake_minimum_required(VERSION 2.8)

option(USE_OPENCV "use opencv for image read/write" ON)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wno-deprecated-gpu-targets -O3")

include_directories(.)
include_directories(../aux)

if (USE_OPENCV)
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  add_definitions(-DUSE_OPENCV)
  set(LIB_LIST "${OpenCV_LIBS}")
endif ()

# utils
set(TOOLS_DIRECTORY ../tools)
include_directories(${TOOLS_DIRECTORY}/cpp)

add_executable(AUX main.cpp ../aux/aux.cpp ${TOOLS_DIRECTORY}/cpp/utils.cpp)
target_link_libraries(AUX ${LIB_LIST})
