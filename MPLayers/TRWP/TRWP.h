#ifndef __TRWP_H__
#define __TRWP_H__

#include <math.h>
#include <stdio.h>
#include <iostream>

#if TORCH_VERSION_MAJOR == 0
  #include <torch/torch.h>  // pytorch 0.4.1, cuda 8
#else
 #include <torch/extension.h>  // pytorch 1.1.0, cuda 10
#endif

#ifdef USE_CUDA
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include "cudaUtils.hpp"
#endif

#include "utils.h"
#include "common.h"

#ifdef __cplusplus
  extern "C" {
#endif

void Horizontal(const Param param,
                float* unary,
                float* msg,
                float* context,
                float* edge_weights);

void DiagonalNarrow(const Param param,
                    float* unary,
                    float* msg,
                    float* context,
                    float* edge_weights);

void DiagonalWide(const Param param,
                  float* unary,
                  float* msg,
                  float* context,
                  float* edge_weights);

void ForwardCPU(const float rho,
                const int n_iter,
                const at::Tensor unary,
                const at::Tensor context,
                const at::Tensor edge_weights,
                at::Tensor msg,
                at::Tensor cost_final,
                at::Tensor label_all);

#ifdef USE_CUDA
void ForwardCUDA(const float rho,
                 const int n_iter,
                 const bool enable_min_a_dir,
                 const at::Tensor unary,
                 const at::Tensor context,
                 const at::Tensor edge_weights,
                 at::Tensor msg,
                 at::Tensor cost_final,
                 at::Tensor msg_min_index,
                 at::Tensor msg_norm_index,
                 at::Tensor unary_update,
                 at::Tensor label_all);

void BackwardCUDA(const float rho,
                  const at::Tensor context,
                  const at::Tensor edge_weights,
                  const at::Tensor dcost_final,
                  const at::Tensor msg_min_index,
                  const at::Tensor msg_norm_index,
                  at::Tensor dunary,
                  at::Tensor dcontext,
                  at::Tensor dedge_weights,
                  at::Tensor dmsg,
                  at::Tensor dunary_update);
#endif

void Forward(const float rho,
             const int n_iter,
             const int enable_min_a_dir,
             const at::Tensor unary,
             const at::Tensor context,
             const at::Tensor edge_weights,
             at::Tensor msg,
             at::Tensor cost_final,
             at::Tensor msg_min_index,
             at::Tensor msg_norm_index,
             at::Tensor unary_update,
             at::Tensor label_all);

void Backward(const float rho,
              const at::Tensor context,
              const at::Tensor edge_weights,
              const at::Tensor dcost_final,
              const at::Tensor msg_min_index,
              const at::Tensor msg_norm_index,
              at::Tensor dunary,
              at::Tensor dcontext,
              at::Tensor dedge_weights,
              at::Tensor dmsg,
              at::Tensor dunary_update);

#ifdef __cplusplus
  }
#endif

#endif  // __TRWP_H__
