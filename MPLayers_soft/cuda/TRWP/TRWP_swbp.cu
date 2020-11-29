#include "TRWP.h"
#include "commonCUDA.cuh"

#ifdef __cplusplus
  extern "C" {
#endif

// =============================================================================
__device__ inline float findMsgMin(const uchar n_disp,
                                   const uchar front_d,
                                   const uchar current_d,
                                   float msg) {
  static __shared__ float min_msg_groups[WARP_SIZE];

  uint warp_id = threadIdx.x / WARP_SIZE;
  min_msg_groups[warp_id] = INFINITY;
  uint n_warps_a_disp = (n_disp + WARP_SIZE - 1) / WARP_SIZE;
  float min_value = msg;
  __syncthreads();

  // Stage 1
  for (uint offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    float friend_value = __shfl_down_sync(FULL_MASK, min_value, offset, WARP_SIZE);
    float min_value_tmp = min(min_value, friend_value);
    min_value = (front_d + offset < n_disp) ? min_value_tmp : min_value;
  }

  min_msg_groups[warp_id] = min_value;
  __syncthreads();

  // Stage 2
  uint offset = current_d * n_warps_a_disp;
  min_value = INFINITY;

  for (uint i = offset; i < offset + n_warps_a_disp; ++i)
    min_value = min(min_value, min_msg_groups[i]);

  __syncthreads();

  return min_value;
}

// =============================================================================
__device__ inline float sumMsg(const uchar n_disp,
                               const uchar current_d,
                               float msg) {
  static __shared__ float sum_msg_groups[WARP_SIZE];

  uint warp_id = threadIdx.x / WARP_SIZE;
  sum_msg_groups[warp_id] = 0;
  uint n_warps_a_disp = (n_disp + WARP_SIZE - 1) / WARP_SIZE;
  float sum_value = msg;
  __syncthreads();

  // Stage 1
  for (uint offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    sum_value += __shfl_down_sync(FULL_MASK, sum_value, offset, WARP_SIZE);

  sum_msg_groups[warp_id] = sum_value;
  __syncthreads();

  // Stage 2
  uint offset = current_d * n_warps_a_disp;
  sum_value = 0;

  for (uint i = offset; i < offset + n_warps_a_disp; ++i)
    sum_value += sum_msg_groups[i];

  __syncthreads();

  return sum_value;
}

// =============================================================================
__device__ inline float findMsgMinIndex(const uchar n_disp,
                                        const float msg,
                                        uchar* msg_min_index) {
  static __shared__ float min_msg_groups[WARP_SIZE];
  static __shared__ unsigned char min_msg_ind_shared[MAX_DISPARITY];
  static __shared__ unsigned char min_msg_ind_groups[WARP_SIZE];

  uchar current_d = threadIdx.x;
  uint warp_id = current_d / WARP_SIZE;
  min_msg_groups[warp_id] = INFINITY;
  min_msg_ind_shared[current_d] = current_d;  // must assign to shared mem for __shfl
  uint n_warps_required = (n_disp + WARP_SIZE - 1) / WARP_SIZE;  // at most 255/32=8
  float min_value = msg;
  uchar min_index = min_msg_ind_shared[current_d];
  __syncthreads();

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

  min_msg_groups[warp_id] = min_value;
  min_msg_ind_groups[warp_id] = min_index;
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
  __syncthreads();

  *msg_min_index = min_index;
  return min_value;
}

// =============================================================================
__global__ void SoftWeightedMessageKernel(const uchar n_disp,
                                          const uint n_thread_required,
                                          const uint n_thread_a_tree,
                                          const float* msg_ptr,
                                          float* msg_soft_sum_norm_ptr,
                                          uchar* msg_soft_sum_min_ind_ptr) {
  static __shared__ float msg_shared[MAX_SHARED_MEM_PER_BLOCK];
  static __shared__ float msg_soft_sum_shared[MAX_DISPARITY];

  uchar n_disp_with_warp = (n_disp + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
  uchar max_parallel_disps = min(n_disp, blockDim.x / n_disp_with_warp);
  uint n_iters = (n_disp + max_parallel_disps - 1) / max_parallel_disps;
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*n_trees*n_thread_a_tree

  if (tid >= n_thread_required) return;

  uchar current_d_base = threadIdx.x / n_disp_with_warp;
  uchar front_d = threadIdx.x % n_disp_with_warp;
  uint tree_id = tid / n_thread_a_tree;
  uint msg_base = tree_id * n_disp * n_disp;
  uint msg_sum_base = tree_id * n_disp;

  // If n_disp is too large, only max_parallel_disps (<= n_disp) can run at one time
  for (uint iter = 0; iter < n_iters; ++iter) {
    uchar current_d = iter * max_parallel_disps + current_d_base;
    uint io_msg_id = msg_base + front_d * n_disp + current_d;
    msg_shared[threadIdx.x] = 0;
    __syncthreads();

    if ((front_d < n_disp) && (current_d < n_disp)) {
      float msg_org = msg_ptr[io_msg_id];
      float msg = msg_org;
      msg_shared[threadIdx.x] = msg;
      __syncthreads();

      msg -= findMsgMin(n_disp, front_d, current_d_base, msg_shared[threadIdx.x]);
      float msg_exp = expf(-msg);
      msg_shared[threadIdx.x] = msg_exp;
      __syncthreads();

      float msg_prob = msg_exp / sumMsg(n_disp, current_d_base, msg_shared[threadIdx.x]);
      float msg_soft = msg_org * msg_prob;
      msg_shared[threadIdx.x] = msg_soft;
      __syncthreads();

      float msg_soft_sum = sumMsg(n_disp, current_d_base, msg_shared[threadIdx.x]);

      msg_soft_sum_shared[current_d] = msg_soft_sum;
    }
    __syncthreads();
  }

  // Norm by subtracting the min soft_msg_sum, at most 255 labels in 1024 threads
  if (threadIdx.x < n_disp) {
    float msg_soft_sum = msg_soft_sum_shared[threadIdx.x];
    uchar msg_soft_sum_min_ind = 0;
    float msg_soft_sum_norm = msg_soft_sum - findMsgMinIndex(n_disp, msg_soft_sum_shared[threadIdx.x], &msg_soft_sum_min_ind);
    msg_soft_sum_norm_ptr[msg_sum_base + threadIdx.x] = msg_soft_sum_norm;
    msg_soft_sum_min_ind_ptr[tree_id] = msg_soft_sum_min_ind;
  }
  __syncthreads();
}

// =============================================================================
__global__ void SoftWeightedMessageBackKernel(const uchar n_disp,
                                              const uint n_thread_required,
                                              const uint n_thread_a_tree,
                                              const float* dmsg_soft_sum_norm_ptr,
                                              const float* msg_ptr,
                                              const uchar* msg_soft_sum_min_ind_ptr,
                                              float* dmsg_ptr) {
  static __shared__ float msg_shared[MAX_SHARED_MEM_PER_BLOCK];
  static __shared__ float dmsg_soft_sum_norm_shared[MAX_DISPARITY];

  uchar n_disp_with_warp = (n_disp + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
  uchar max_parallel_disps = min(n_disp, blockDim.x / n_disp_with_warp);
  uint n_iters = (n_disp + max_parallel_disps - 1) / max_parallel_disps;
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*n_trees*n_thread_a_tree

  if (tid >= n_thread_required) return;

  uchar current_d_base = threadIdx.x / n_disp_with_warp;
  uchar front_d = threadIdx.x % n_disp_with_warp;
  uint tree_id = tid / n_thread_a_tree;
  uint msg_base = tree_id * n_disp * n_disp;
  uint msg_sum_base = tree_id * n_disp;

  // ===== Back norm, TODO: not that good but by far use it
  uchar msg_soft_sum_min_ind = msg_soft_sum_min_ind_ptr[tree_id];

  if (threadIdx.x < n_disp) {
    dmsg_soft_sum_norm_shared[threadIdx.x] = dmsg_soft_sum_norm_ptr[msg_sum_base + threadIdx.x];
  }
  __syncthreads();

  float sum = 0;
  if (threadIdx.x == 0) {
    for (uint i = 0; i < n_disp; ++i) {
      sum += dmsg_soft_sum_norm_shared[i];
    }
    dmsg_soft_sum_norm_shared[msg_soft_sum_min_ind] -= sum;
  }
  __syncthreads();
  // dmsg_soft_sum_norm_shared[msg_soft_sum_min_ind] -= sumMsg(n_disp, 0, dmsg_soft_sum_norm_shared[current_d_base]);

  for (uint iter = 0; iter < n_iters; ++iter) {
    uchar current_d = iter * max_parallel_disps + current_d_base;

    if ((front_d < n_disp) && (current_d < n_disp)) {
      uint io_msg_id = msg_base + front_d * n_disp + current_d;

      // Cal msg_soft_sum
      float msg_org = msg_ptr[io_msg_id];
      msg_shared[threadIdx.x] = msg_org;
      __syncthreads();

      float msg = msg_org - findMsgMin(n_disp, front_d, current_d_base, msg_shared[threadIdx.x]);
      float msg_exp = expf(-msg);
      msg_shared[threadIdx.x] = msg_exp;
      __syncthreads();

      float msg_prob = msg_exp / sumMsg(n_disp, current_d_base, msg_shared[threadIdx.x]);
      msg_shared[threadIdx.x] = msg_prob * msg_org;
      __syncthreads();

      float msg_soft_sum = sumMsg(n_disp, current_d_base, msg_shared[threadIdx.x]);

      dmsg_ptr[io_msg_id] = dmsg_soft_sum_norm_shared[current_d] * msg_prob * (1 - msg_org + msg_soft_sum);
      __syncthreads();
    }
    __syncthreads();
  }
}

// =============================================================================
void SoftWeightedMessageCUDA(const at::Tensor message,
                             at::Tensor message_soft_sum_norm,
                             at::Tensor message_soft_sum_min_ind) {
  // Message:(n_trees,n_disp,n_disp)
  const uint n_trees = message.size(0);
  const uchar n_disp = message.size(1);
  float* msg_ptr = message.data<float>();
  float* msg_soft_sum_norm_ptr = message_soft_sum_norm.data<float>();
  uchar* msg_soft_sum_min_ind_ptr = message_soft_sum_min_ind.data<uchar>();

  uchar n_disp_with_warp = GetNumThreadATree(n_disp, WARP_SIZE);
  uint n_thread_a_tree = min(n_disp, MAX_THREADS_PER_BLOCK / n_disp_with_warp) * n_disp_with_warp;
  uint n_thread_required = n_trees * n_thread_a_tree;
  uint n_block_soft= (n_thread_required + n_thread_a_tree - 1) / n_thread_a_tree;

  SoftWeightedMessageKernel<<<n_block_soft, n_thread_a_tree>>>(n_disp,
                                                               n_thread_required,
                                                               n_thread_a_tree,
                                                               msg_ptr,
                                                               msg_soft_sum_norm_ptr,
                                                               msg_soft_sum_min_ind_ptr);
}

// =============================================================================
void SoftWeightedMessageBackCUDA(const at::Tensor dmessage_soft_sum_norm,
                                 const at::Tensor message,
                                 const at::Tensor message_soft_sum_min_ind,
                                 at::Tensor dmessage) {
  const uint n_trees = message.size(0);
  const uchar n_disp = message.size(1);
  float* dmsg_soft_sum_norm_ptr = dmessage_soft_sum_norm.data<float>();
  float* msg_ptr = message.data<float>();
  uchar* msg_soft_sum_min_ind_ptr = message_soft_sum_min_ind.data<uchar>();
  float* dmsg = dmessage.data<float>();

  uchar n_disp_with_warp = GetNumThreadATree(n_disp, WARP_SIZE);
  uint n_thread_a_tree = min(n_disp, MAX_THREADS_PER_BLOCK / n_disp_with_warp) * n_disp_with_warp;
  uint n_thread_required = n_trees * n_thread_a_tree;
  uint n_block_soft = (n_thread_required + n_thread_a_tree - 1) / n_thread_a_tree;

  SoftWeightedMessageBackKernel<<<n_block_soft, n_thread_a_tree>>>(n_disp,
                                                                   n_thread_required,
                                                                   n_thread_a_tree,
                                                                   dmsg_soft_sum_norm_ptr,
                                                                   msg_ptr,
                                                                   msg_soft_sum_min_ind_ptr,
                                                                   dmsg);
}

#ifdef __cplusplus
  }
#endif
