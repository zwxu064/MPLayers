#ifndef __TRWP_SOFT_H__
#define __TRWP_SOFT_H__

#include "utils.h"
#include "TRWP.h"

#ifdef __cplusplus
  extern "C" {
#endif

// =============================================================================
__device__ inline float findMsgMin(const uchar n_disp,
                                   const uchar front_d,
                                   const uchar current_d,
                                   float msg) {
  static __shared__ float min_msg_groups[WARP_SIZE];
  if (threadIdx.x < WARP_SIZE) min_msg_groups[threadIdx.x] = INFINITY;
  __syncthreads();

  uint warp_id = threadIdx.x / WARP_SIZE;
  uint n_warps_a_disp = (n_disp + WARP_SIZE - 1) / WARP_SIZE;
  float min_value = msg;

  // Stage 1
  for (uint offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float friend_value = __shfl_down_sync(FULL_MASK, min_value, offset, WARP_SIZE);
    float min_value_tmp = min(min_value, friend_value);
    min_value = (front_d + offset < n_disp) ? min_value_tmp : min_value;
  }

  if (threadIdx.x % WARP_SIZE == 0) min_msg_groups[warp_id] = min_value;
  __syncthreads();

  // Stage 2
  uint offset = current_d * n_warps_a_disp;
  min_value = INFINITY;

  for (uint i = offset; i < offset + n_warps_a_disp; ++i)
    min_value = min(min_value, min_msg_groups[i]);

  return min_value;
}

// =============================================================================
__device__ inline float sumMsg(const uchar n_disp,
                               const uchar current_d,
                               float msg) {
  static __shared__ float sum_msg_groups[WARP_SIZE];
  if (threadIdx.x < WARP_SIZE) sum_msg_groups[threadIdx.x] = 0;
  __syncthreads();

  uint warp_id = threadIdx.x / WARP_SIZE;
  uint n_warps_a_disp = (n_disp + WARP_SIZE - 1) / WARP_SIZE;
  float sum_value = msg;

  // Stage 1
  for (uint offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    sum_value += __shfl_down_sync(FULL_MASK, sum_value, offset, WARP_SIZE);

  if (threadIdx.x % WARP_SIZE == 0) sum_msg_groups[warp_id] = sum_value;
  __syncthreads();

  // Stage 2
  uint offset = current_d * n_warps_a_disp;
  sum_value = 0;

  for (uint i = offset; i < offset + n_warps_a_disp; ++i)
    sum_value += sum_msg_groups[i];

  return sum_value;
}

// =============================================================================
__device__ inline float findMsgMinIndex(const uchar n_disp,
                                        const float msg,
                                        uchar* msg_min_index) {
  static __shared__ float min_msg_groups[WARP_SIZE];
  static __shared__ uchar min_msg_ind_groups[WARP_SIZE];

  if (threadIdx.x < WARP_SIZE) {
    min_msg_groups[threadIdx.x] = INFINITY;
    min_msg_ind_groups[threadIdx.x] = uchar(0);
  }
  __syncthreads();

  uint n_disp_with_warp = (n_disp + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
  uint n_warps_required = (n_disp + WARP_SIZE - 1) / WARP_SIZE;  // at most 255/32=8
  uint current_d = threadIdx.x % n_disp_with_warp; // Was a bug 13rd Nov. 2019: limit this since there would be at most 1024 threads
  uint warp_id = threadIdx.x / WARP_SIZE;

  float min_value = msg;
  uchar min_index = current_d;

  // Stage 1
  for (uint offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float friend_value = __shfl_down_sync(FULL_MASK, min_value, offset, WARP_SIZE);
    uchar friend_index = __shfl_down_sync(FULL_MASK, min_index, offset, WARP_SIZE);

    if (current_d + offset < n_disp) {
#if TORCH_VERSION_MAJOR == 0
      bool enable_index_swap = (friend_value < min_value) || ((friend_value == min_value) && (friend_index < min_index));
#else
      bool enable_index_swap = (friend_value < min_value) || ((friend_value == min_value) && (friend_index > min_index));
#endif

      min_value = min(friend_value, min_value);
      min_index = enable_index_swap ? friend_index : min_index;
    }
    __syncthreads();
  }

  if (threadIdx.x % WARP_SIZE == 0) {
    min_msg_groups[warp_id] = min_value;
    min_msg_ind_groups[warp_id] = min_index;
  }
  __syncthreads();

  // Stage 2
  min_value = min_msg_groups[0];
  min_index = min_msg_ind_groups[0];

  for (uint i = 1; i < n_warps_required; ++i) {
    float current_value = min_msg_groups[i];
    uchar current_index = min_msg_ind_groups[i];

  #if TORCH_VERSION_MAJOR == 0
    bool enable_index_swap = (current_value < min_value) || ((current_value == min_value) && (current_index < min_index));
  #else
    bool enable_index_swap = (current_value < min_value) || ((current_value == min_value) && (current_index > min_index));
  #endif

    min_value = min(current_value, min_value);
    min_index = enable_index_swap ? current_index : min_index;
  }

  *msg_min_index = min_index;

  return min_value;

//  static __shared__ float msg_shared[MAX_DISPARITY];
//  static __shared__ float final_msg_value;
//  static __shared__ uchar final_msg_index;
//
//  if (threadIdx.x < n_disp)
//    msg_shared[threadIdx.x] = msg;
//  else if (threadIdx.x < MAX_DISPARITY)
//    msg_shared[threadIdx.x] = INFINITY;
//  __syncthreads();
//
//  if (threadIdx.x == 0) {
//    float min_value = msg_shared[0];
//    uchar min_index = 0;
//
//    for (uint ii = 1; ii < n_disp; ++ii) {
//      float friend_value = msg_shared[ii];
//
//      if (friend_value <= min_value) {
//        min_index = ii;
//        min_value = friend_value;
//      }
//    }
//    final_msg_value = min_value;
//    final_msg_index = min_index;
//  }
//  __syncthreads();
//
//  *msg_min_index = final_msg_index;
//
//  return final_msg_value;
}

#ifdef __cplusplus
  }
#endif

#endif  // __TRWP_SOFT_H__
