cmake_minimum_required(VERSION 2.8.0)
project(GPU_inf)

set(CMAKE_XX_FLAGS "-std=c++11")
find_package(CUDA)

include_directories(/usr/local/cuda-10.0/include)
link_directories(/usr/local/cuda-10.0/lib64)

cuda_add_executable(GPU_inf main.cpp)
target_link_libraries(GPU_inf -lcudart)
