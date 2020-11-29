#include "cudaUtils.hpp"

void __CUDAErrorCheck(const char *file,
                      const int line) {
    cudaError status = cudaGetLastError();
    if (cudaSuccess != status) {
        printf("!!!Error, CUDAErrorCheck failed, %s:%i : %s\n",
               file, line, cudaGetErrorString(status));
        exit(-1);
    }

    status = cudaDeviceSynchronize();
    if(cudaSuccess != status) {
        printf("!!!Error, CUDAErrorCheck sync failed, %s:%i : %s\n",
               file, line, cudaGetErrorString(status));
        exit(-1);
    }
}
