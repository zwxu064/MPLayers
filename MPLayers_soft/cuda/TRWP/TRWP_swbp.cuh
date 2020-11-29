#include "utils.h"
#include "TRWP.h"

#ifdef __cplusplus
  extern "C" {
#endif

__global__ void SoftWeightedMessageKernel(const uchar n_disp,
                                          const uint n_thread_required,
                                          const uint n_thread_a_tree,
                                          const float* msg_ptr,
                                          float* msg_soft_sum_norm_ptr,
                                          uchar* msg_soft_sum_min_ind_ptr);

__global__ void SoftWeightedMessageBackKernel(const uchar n_disp,
                                              const uint n_thread_required,
                                              const uint n_thread_a_tree,
                                              const float* dmsg_soft_sum_norm_ptr,
                                              const float* msg_ptr,
                                              const uchar* msg_soft_sum_min_ind_ptr,
                                              float* msg_grad_ptr);

void SoftWeightedMessageCUDA(const at::Tensor message,
                             at::Tensor message_soft_sum_norm,
                             at::Tensor message_soft_sum_min_ind);

void SoftWeightedMessageBackCUDA(const at::Tensor dmessage_soft_sum_norm,
                                 const at::Tensor message,
                                 const at::Tensor message_soft_sum_min_ind,
                                 at::Tensor dmessage);

#ifdef __cplusplus
  }
#endif
