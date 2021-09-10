#ifndef __ISGMR_H__
#define __ISGMR_H__

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
                float* edge_weights,
                float* msg_update);

void DiagonalNarrow(const Param param,
                    float* unary,
                    float* msg,
                    float* context,
                    float* edge_weights,
                    float* msg_update);

void DiagonalWide(const Param param,
                  float* unary,
                  float* msg,
                  float* context,
                  float* edge_weights,
                  float* msg_update);

void ForwardCPU(const bool enable_sgm,
                const int sgm_single_mode,
                const float rho,
                const int n_iter,
                const at::Tensor unary,
                const at::Tensor context,
                const at::Tensor edge_weights,
                at::Tensor msg,
                at::Tensor cost_final,
                at::Tensor msg_update,
                at::Tensor label_all);

#ifdef USE_CUDA
void ForwardCUDA(const bool enable_sgm,
                 const int sgm_single_mode,
                 const float rho,
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
                 at::Tensor msg_update,
                 at::Tensor label_all);

void BackwardCUDA(const bool enable_sgm,
                  const int sgm_single_mode,
                  const float rho,
                  const at::Tensor context,
                  const at::Tensor edge_weights,
                  const at::Tensor dcost_final,
                  const at::Tensor msg_min_index,
                  const at::Tensor msg_norm_index,
                  at::Tensor dunary,
                  at::Tensor dcontext,
                  at::Tensor dedge_weights,
                  at::Tensor dmsg,
                  at::Tensor dunary_update,
                  at::Tensor dmsg_update);
#endif

void Forward(const int enable_sgm,
             const int sgm_single_mode,
             const float rho,
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
             at::Tensor msg_update,
             at::Tensor label_all);

void Backward(const int enable_sgm,
              const int sgm_single_mode,
              const float rho,
              const at::Tensor context,
              const at::Tensor edge_weights,
              const at::Tensor dcost_final,
              const at::Tensor msg_min_index,
              const at::Tensor msg_norm_index,
              at::Tensor dunary,
              at::Tensor dcontext,
              at::Tensor dedge_weights,
              at::Tensor dmsg,
              at::Tensor dunary_update,
              at::Tensor dmsg_update);

// Extra

#ifdef USE_CUDA
void TestMsgNormCUDA(at::Tensor msg_norm);
#endif

void TestMsgNorm(at::Tensor msg_norm);

#ifdef USE_CUDA
void TestMultiStreamCUDA(const int enable_multiple,
                         at::Tensor data);
#endif

void TestMultiStream(const int enable_multiple,
                     at::Tensor data);

#ifdef __cplusplus
  }
#endif

#endif  // __ISGMR_H__
