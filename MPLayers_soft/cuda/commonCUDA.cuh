#ifndef __COMMONCUDA_H__
#define __COMMONCUDA_H__

// Reference from https://stackoverflow.com/questions/6252912/cuda-function-call-from-anther-cu-file

#pragma once

#ifdef __cplusplus
  extern "C" {
#endif

// =============================================================================
using clock_value_t = long long;

__device__ inline void CuSleep(clock_value_t sleep_cycles)
{
  clock_value_t start = clock64();
  clock_value_t cycles_elapsed;
  do { cycles_elapsed = clock64() - start; }
  while (cycles_elapsed < sleep_cycles);
}

// =============================================================================
// 1. offset=(offset+1)/2 is for 2^* diviable n_disp;
// 2. return index corresponding to the min value.
// Mask is useful for SumReduction but not for Min/Max-Reduction
// PyTorch 1.1.0 uses max index for min() and max()
// Shuffling sometime swaps the indices when the two values are equal, so use max or min
__device__ inline void SoftWarpReduceMin(const uint n_disp,
                                         const uint current_d,
                                         float value,
                                         float* out_value,
                                         uchar* out_idx) {
  uint group = current_d / WARP_SIZE;
  uint n_groups = (n_disp + WARP_SIZE - 1) / WARP_SIZE;

  uint lane_id = current_d % WARP_SIZE;
  uint offset_start = WARP_SIZE;
  uint mod_value = n_disp % WARP_SIZE;
  if ((mod_value != 0) && (group == n_groups - 1)) offset_start = mod_value;
  uint offset_his = offset_start;
  float min_value = value;
  uint min_idx = current_d;

  for (uint offset = (offset_start + 1) / 2; offset > 0; offset = (offset + 1) / 2) {
    if (offset_his >= 2) {
      uint offset_use = ((offset_his % 2 == 1) && (lane_id == offset - 1)) ? 0 : offset;
      float value_friend = __shfl_down_sync(FULL_MASK, min_value, offset_use, WARP_SIZE);
      uchar idx_friend = __shfl_down_sync(FULL_MASK, min_idx, offset_use, WARP_SIZE);

      if (value_friend <= min_value) {
        min_idx = (value_friend == min_value) ? max(min_idx, idx_friend) : idx_friend;
        min_value = value_friend;
      }

      offset_his = offset;  // to see if the number of next available lanes is odd or even
    }
  }

  *out_value = min_value;
  *out_idx = (uchar)min_idx;
}

// =============================================================================
__device__ inline void MsgNorm(const uint n_disp,
                               const uint current_d,
                               float* smem,
                               uchar* msg_norm_index) {
  static __shared__ float min_value_shared[MAX_WARP_GROUP];
  static __shared__ uchar min_idx_shared[MAX_WARP_GROUP];
  static __shared__ float final_min_value;
  static __shared__ uchar final_min_idx;

  uint n_groups = (n_disp + WARP_SIZE - 1) / WARP_SIZE;
  uint group = current_d / WARP_SIZE;
  SoftWarpReduceMin(n_disp, current_d, smem[current_d], &min_value_shared[group],
                    &min_idx_shared[group]);
  __syncthreads();

  if (current_d < n_groups)
    SoftWarpReduceMin(n_groups, current_d, min_value_shared[current_d],
                      &final_min_value, &final_min_idx);
  __syncthreads();

  smem[current_d] -= final_min_value;
  *msg_norm_index = min_idx_shared[final_min_idx];
}

// =============================================================================
// This one is faster may be because SoftMsgNorm has two shfl and also
// handle non 2^* dividable disparity number
__device__ inline void MsgNormNaive(const uint n_disp,
                                    const uint smem_id,
                                    float* smem,
                                    uchar* msg_norm_index) {
  static __shared__ float min_shared;
  static __shared__ uchar min_shared_idx;

  int disp = smem_id % n_disp;

  if (disp == 0) {
#if TORCH_VERSION_MAJOR == 0
    for (int d = 0; d < n_disp; ++d) {
      float value = smem[d];
      if (d == 0) {
        min_shared = value;
        min_shared_idx = d;
      } else if (value < min_shared) {
        min_shared = value;
        min_shared_idx = d;
      }
    }
#else
    for (int d = n_disp - 1; d >= 0; --d) {
      float value = smem[d];
      if (d == n_disp - 1) {
        min_shared = value;
        min_shared_idx = d;
      } else if (value < min_shared) {
        min_shared = value;
        min_shared_idx = d;
      }
    }
#endif
  }
  __syncthreads();
  smem[smem_id] -= min_shared;
  *msg_norm_index = min_shared_idx;
}

// =============================================================================
__device__ inline void MsgNormNaiveBack(const uint n_disp,
                                        const uint smem_id,
                                        const uint msg_norm_index,
                                        float* dmsg_update) {
  uint disp = smem_id % n_disp;

  if (disp == 0) {
    float dmsg_update_sum = 0;
    for (uint d = 0; d < n_disp; ++d)
      dmsg_update_sum += dmsg_update[d];
    atomicAdd(&dmsg_update[msg_norm_index], -dmsg_update_sum);
  }

  __syncthreads();
}

#ifdef __cplusplus
  }
#endif

#endif  // __COMMONCUDA_H__
