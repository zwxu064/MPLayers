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

#if defined(USE_CUDA) && defined(USE_SOFT)
  #include "TRWP_swbp.cuh"
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

#if defined(USE_CUDA) && defined(USE_HARD)
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
#endif

#if defined(USE_CUDA) && defined(USE_SOFT)
void ForwardCUDASoft(const float rho,
                     const int n_iter,
                     const at::Tensor unary,
                     const at::Tensor context,
                     const at::Tensor edge_weights,
                     at::Tensor msg,
                     at::Tensor msg_edge_label,
                     at::Tensor msg_norm_index,
                     at::Tensor cost_final,
                     at::Tensor unary_update,
                     at::Tensor label_all);

void BackwardCUDASoft(const float rho,
                      const at::Tensor dcost_final,
                      const at::Tensor context,
                      const at::Tensor edge_weights,
                      const at::Tensor msg_edge_label,
                      const at::Tensor msg_norm_index,
                      at::Tensor dunary,
                      at::Tensor dcontext,
                      at::Tensor dedge_weights,
                      at::Tensor dmsg,
                      at::Tensor dunary_update);

// msg:(n_dir, batch, cv, h, w, n_disp)
// cost_final:(batch, cv, h, w, n_disp)
// msg_edge_label:(n_iter, n_dir, batch, cv, h, w, n_disp, n_disp)
// msg_norm_index:(n_iter, n_dir, batch, cv, h, w)
void ForwardSoft(const float rho,
                 const int n_iter,
                 const at::Tensor unary,
                 const at::Tensor context,
                 const at::Tensor edge_weights,
                 at::Tensor msg,
                 at::Tensor msg_edge_label,
                 at::Tensor msg_norm_index,
                 at::Tensor cost_final,
                 at::Tensor unary_update,
                 at::Tensor label_all);

void BackwardSoft(const float rho,
                  const at::Tensor dcost_final,
                  const at::Tensor context,
                  const at::Tensor edge_weights,
                  const at::Tensor msg_edge_label,
                  const at::Tensor msg_norm_index,
                  at::Tensor dunary,
                  at::Tensor dcontext,
                  at::Tensor dedge_weights,
                  at::Tensor dmsg,
                  at::Tensor dunary_update);

void SoftWeightedMessage(const at::Tensor message,
                         at::Tensor message_soft_sum_norm,
                         at::Tensor message_soft_sum_min_ind);

void SoftWeightedMessageBack(const at::Tensor dmessage_soft_sum_norm,
                             const at::Tensor message,
                             const at::Tensor message_soft_sum_min_ind,
                             at::Tensor dmessage_grad);
#endif

#ifdef __cplusplus
  }
#endif

#endif  // __TRWP_H__
