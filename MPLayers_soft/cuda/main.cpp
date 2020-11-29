#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// From http://horacio9573.no-ip.org/cuda/structcudaDeviceProp.html
int main() {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << "Name: " << prop.name << std::endl;
    std::cout << "MaxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "MaxThreadDim: " << *prop.maxThreadsDim << std::endl;
    std::cout << "MaxGridSize: " << prop.maxGridSize[0]
              << ", " << prop.maxGridSize[1]
              << ", " << prop.maxGridSize[2] << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "SharedMemoryPerBlock: " << prop.sharedMemPerBlock / 1024 << "KB" << std::endl;
    std::cout << "\n" << std::endl;
  }
}
