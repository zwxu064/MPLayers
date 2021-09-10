#include "ISGMR.h"
#include "commonCUDA.cuh"

/*
 * Note
 * Bug 1: 20190803, PyTorch 0.4.1, torch.max and torch.min return min_indices
 *        while pytorch 1.1.0 return max_indices, using PyTorch 1.1.0 now
*/

#ifdef __cplusplus
  extern "C" {
#endif

__global__ void TestLoop(const Param param,
                         int required,
                         float* out) {
  int n_disp = param.n_disp;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= required) return;

  int value = 99;
  for (int d = 0; d < n_disp; ++d) {
    if (d == 0) {
      value = 2;
//      out[tid] = 2;
    } else if (d == 15) {
      value = 15;
    }
//    printf("value=%d.\n", value);
  }
  out[tid] = value;
}

__global__ void CalLabelKernel(const Param param,
                               const uint n_thread_required,
                               float* cost_final,
                               uchar* label_all) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_thread_required) return;

  uint n_disp = param.n_disp;
  float min_value = cost_final[tid * n_disp];
  uchar min_index = 0;

  for (uint disp = 1; disp < n_disp; ++disp) {
    float value = cost_final[tid * n_disp + disp];

    if (value < min_value) {
      min_value = value;
      min_index = disp;
    }
  }

  label_all[tid] = min_index;
}

__device__ float MsgSumAll(const int n_dir,
                           const int dir,
                           const int msg_dir_size,
                           const int msg_offset,
                           float* msg) {
  float msg_sum = 0;
  for (int d = 0; d < n_dir; ++d)
    if (d != dir)
      msg_sum += msg[d * msg_dir_size + msg_offset];
  return msg_sum;
}

__global__ void CostAggregateKernel(const bool enable_sgm,
                                    const Param param,
                                    const uint n_thread_required,
                                    float* msg_ptr,
                                    float* cost_final_ptr) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_thread_required) return;

  uint msg_offset = n_thread_required, n_dir = param.n_dir;
  float value_factor = (float)1;
  if (enable_sgm) value_factor = (float)n_dir;
  float value = cost_final_ptr[tid] * value_factor;

  if (param.enable_min_a_dir) {
    float min_msg = msg_ptr[tid];

    for (uint dir = 1; dir < n_dir; ++dir) {
      float msg = msg_ptr[dir * msg_offset + tid];
      if (msg < min_msg) min_msg = msg;
    }

    value += min_msg;
  } else {
    for (uint dir = 0; dir < n_dir; ++dir) {
      value += msg_ptr[dir * msg_offset + tid] / param.dir_weight;
    }
  }

  cost_final_ptr[tid] = value;
  __syncthreads();
}

__global__ void UpdateUnaryKernel(const Param param,
                                  int n_thread_required,
                                  float* unary_ptr,
                                  float* msg_ptr,
                                  float* unary_update_ptr) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_thread_required) return;

  uint msg_offset = n_thread_required;
  uint dir_current = param.dir, dir_inv = param.dir_inv, n_dir = param.n_dir;
  float rho = param.rho, value = unary_ptr[tid];

  for (uint dir = 0; dir < n_dir; ++dir)
    value += msg_ptr[dir * msg_offset + tid];

  value -= msg_ptr[dir_current * msg_offset + tid];
  value *= rho;
  value -= msg_ptr[dir_inv * msg_offset + tid];
  unary_update_ptr[tid] = value;
  __syncthreads();
}

__global__ void HorizontalKernel(const Param param,
                                 const uint n_thread_required,
                                 const uint n_thread_a_tree,
                                 float* unary_update,
                                 float* context,
                                 float* edge_weights,
                                 float* msg,
                                 float* msg_update,
                                 uchar* msg_min_index,
                                 uchar* msg_norm_index) {
  static __shared__ float msg_update_shared[MAX_DISPARITY];
  msg_update_shared[threadIdx.x] = 0;
  __syncthreads();

  uint height = param.height, width = param.width;
  uint n_disp = param.n_disp, n_trees = param.n_trees;
  float rho = param.rho;
  int h_step = param.h_step, w_step = param.w_step;
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*h*n_thread_a_tree
  uint current_d = threadIdx.x % n_thread_a_tree;
  bool enable_seg = (n_disp == 21);

  if (tid >= n_thread_required) return;
  if (current_d >= n_disp) return;

  uint unary_base = tid / (n_trees * n_thread_a_tree) * height * width * n_disp;
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
  int h_start = tree_id, w_start = (w_step > 0) ? 0 : (width - 1);
  uint edge_base = tid / (n_trees * n_thread_a_tree) * height * width;

  for (uint i = 0; i < width; ++i) {
    int current_node_h = h_start;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h - h_step;
    int front_node_w = current_node_w - w_step;

    if (0 <= current_node_w && current_node_w < width &&
        0 <= front_node_w && front_node_w < width) {
      float min_value = 0;
      uchar min_idx = 0;
      uint offset_base = unary_base + front_node_h * width * n_disp + front_node_w * n_disp;
      float edge_weight = edge_weights[edge_base + current_node_h * width + current_node_w];

#if TORCH_VERSION_MAJOR == 0
      for (int front_d = 0; front_d < n_disp; ++front_d) {
#else
      for (int front_d = n_disp - 1; front_d >= 0; --front_d) {
#endif
        float context_value = 0;
        if (enable_seg)
         context_value = context[min(current_d, front_d) * n_disp + max(current_d, front_d)];
        else
         context_value = context[std::abs(int(current_d) - int(front_d))];

        uint offset = offset_base + front_d;
        float msg_update_value = rho * msg_update_shared[front_d];
        __syncthreads();
        float value = unary_update[offset] + msg_update_value + edge_weight * context_value;

#if TORCH_VERSION_MAJOR == 0
        if (front_d == 0) {
#else
        if (front_d == n_disp - 1) {
#endif
          min_value = value;
          min_idx = front_d;
        } else if (value < min_value) {
          min_value = value;
          min_idx = front_d;
        }
      }

      msg_update_shared[current_d] = min_value;
      __syncthreads();

      int msg_offset = unary_base + current_node_h * width * n_disp + current_node_w * n_disp + current_d;
      int msg_index_offset = tid / n_thread_a_tree * width + current_node_w;
      uchar norm_idx = 0;

#ifdef USE_MSGNORM_NAIVE
      MsgNormNaive(param.n_disp, current_d, msg_update_shared, &norm_idx);
#else
      MsgNorm(param.n_disp, current_d, msg_update_shared, &norm_idx);
#endif

      msg_update[msg_offset] = msg_update_shared[current_d];

      if (param.is_training) {
        msg_norm_index[msg_index_offset] = norm_idx;
        msg_min_index[msg_offset] = min_idx;
      }
      __syncthreads();
    }
  }
}

__global__ void DiagonalKernelNarrow(const Param param,
                                     const uint n_thread_required,
                                     const uint n_thread_a_tree,
                                     float* unary_update,
                                     float* context,
                                     float* edge_weights,
                                     float* msg,
                                     float* msg_update,
                                     uchar* msg_min_index,
                                     uchar* msg_norm_index) {
  static __shared__ float msg_update_shared[MAX_DISPARITY];
  msg_update_shared[threadIdx.x] = 0;
  __syncthreads();

  uint height = param.height, width = param.width;
  uint n_disp = param.n_disp, n_trees = param.n_trees;
  float rho = param.rho;
  int h_step = param.h_step, w_step = param.w_step;
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*n_trees*n_thread_a_tree
  uint current_d = threadIdx.x % n_thread_a_tree;
  uint h_step_abs = std::abs(h_step);
  bool enable_seg = (n_disp == 21);

  if (tid >= n_thread_required) return;
  if (current_d >= n_disp) return;

  uint unary_base = tid / (n_trees * n_thread_a_tree) * height * width * n_disp;
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
  int tree_id_shift = tree_id - (height - 1) * max(w_step, 0);
  int common1 = tree_id_shift % h_step_abs;
  float common2 = float(tree_id_shift) / float(h_step_abs);  // This must be float NOT int, will affect ceilf and floorf
  int h_start = 0, w_start = 0;
  uint edge_base = tid / (n_thread_a_tree * n_trees) * height * width;

  // Use a common mode to calculate start points for shortest chains, read my notes for clarity
  if (w_step > 0) {
    h_start = (h_step_abs - common1) % h_step_abs;
    w_start = ceilf(common2);
  } else {
    h_start = common1;
    w_start = floorf(common2);
  }

  if (h_step < 0) h_start = height - 1 - h_start;
  uint roll_step = (height - 1) / h_step_abs;

  for (uint i = 0; i <= roll_step; ++i) {
    int current_node_h = h_start + i * h_step;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h - h_step;
    int front_node_w = current_node_w - w_step;

    if (0 <= current_node_h && current_node_h < height &&
        0 <= current_node_w && current_node_w < width &&
        0 <= front_node_h && front_node_h < height &&
        0 <= front_node_w && front_node_w < width) {
      float min_value = 0;
      uchar min_idx = 0;
      uint offset_base = unary_base + front_node_h * width * n_disp + front_node_w * n_disp;
      float edge_weight = edge_weights[edge_base + current_node_h * width + current_node_w];

#if TORCH_VERSION_MAJOR == 0
      for (int front_d = 0; front_d < n_disp; ++front_d) {
#else
      for (int front_d = n_disp - 1; front_d >= 0; --front_d) {
#endif
        float context_value = 0;
        if (enable_seg)
          context_value = context[min(current_d, front_d) * n_disp + max(current_d, front_d)];
        else
          context_value = context[std::abs(int(current_d) - int(front_d))];

        uint offset = offset_base + front_d;
        float msg_update_value = rho * msg_update_shared[front_d];
        __syncthreads();
        float value = unary_update[offset] + msg_update_value + edge_weight * context_value;

#if TORCH_VERSION_MAJOR == 0
        if (front_d == 0) {
#else
        if (front_d == n_disp - 1) {
#endif
          min_value = value;
          min_idx = front_d;
        } else if (value < min_value) {
          min_value = value;
          min_idx = front_d;
        }
      }

      msg_update_shared[current_d] = min_value;
      __syncthreads();

      uint msg_offset = unary_base + current_node_h * width * n_disp + current_node_w * n_disp + current_d;
      uint msg_index_offset = tid / (n_thread_a_tree * n_trees) * height * width + current_node_h * width + current_node_w;
      uchar norm_idx = 0;

#ifdef USE_MSGNORM_NAIVE
      MsgNormNaive(param.n_disp, current_d, msg_update_shared, &norm_idx);
#else
      MsgNorm(param.n_disp, current_d, msg_update_shared, &norm_idx);
#endif

      msg_update[msg_offset] = msg_update_shared[current_d];

      if (param.is_training) {
        msg_norm_index[msg_index_offset] = norm_idx;
        msg_min_index[msg_offset] = min_idx;
      }
      __syncthreads();
    }
  }
}

__global__ void DiagonalKernelWide(const Param param,
                                   const uint n_thread_required,
                                   const uint n_thread_a_tree,
                                   float* unary_update,
                                   float* context,
                                   float* edge_weights,
                                   float* msg,
                                   float* msg_update,
                                   uchar* msg_min_index,
                                   uchar* msg_norm_index) {
  static __shared__ float msg_update_shared[MAX_DISPARITY];
  msg_update_shared[threadIdx.x] = 0;
  __syncthreads();

  uint height = param.height, width = param.width;
  uint n_disp = param.n_disp, n_trees = param.n_trees;
  float rho = param.rho;
  int h_step = param.h_step, w_step = param.w_step;
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*n_trees*n_thread_a_tree
  uint current_d = threadIdx.x % n_thread_a_tree;
  bool enable_seg = (n_disp == 21);

  if (tid >= n_thread_required) return;
  if (current_d >= n_disp) return;

  uint unary_base = tid / (n_trees * n_thread_a_tree) * height * width * n_disp;
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
  int tree_id_shift = tree_id - (height - 1) * max(w_step, 0);
  uint h_step_abs = std::abs(h_step), roll_step = (height - 1) / h_step_abs;
  int h_start = (h_step > 0) ? 0 : (height - 1), w_start = tree_id_shift;
  uint edge_base = tid / (n_thread_a_tree * n_trees) * height * width;

  for (uint i = 0; i <= roll_step; ++i) {
    int current_node_h = h_start + i * h_step;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h - h_step;
    int front_node_w = current_node_w - w_step;

    if (0 <= current_node_h && current_node_h < height &&
        0 <= current_node_w && current_node_w < width &&
        0 <= front_node_h && front_node_h < height &&
        0 <= front_node_w && front_node_w < width) {
      float min_value = 0;
      uchar min_idx = 0;
      uint offset_base = unary_base + front_node_h * width * n_disp + front_node_w * n_disp;
      float edge_weight = edge_weights[edge_base + current_node_h * width + current_node_w];

#if TORCH_VERSION_MAJOR == 0
      for (int front_d = 0; front_d < n_disp; ++front_d) {
#else
      for (int front_d = n_disp - 1; front_d >= 0; --front_d) {
#endif
        float context_value = 0;
        if (enable_seg)
          context_value = context[min(current_d, front_d) * n_disp + max(current_d, front_d)];
        else
          context_value = context[std::abs(int(current_d) - int(front_d))];

        uint offset = offset_base + front_d;
        float msg_update_value = rho * msg_update_shared[front_d];
        __syncthreads();
        float value = unary_update[offset] + msg_update_value + edge_weight * context_value;

#if TORCH_VERSION_MAJOR == 0
        if (front_d == 0) {
#else
        if (front_d == n_disp - 1) {
#endif
          min_value = value;
          min_idx = front_d;
        } else if (value < min_value) {
          min_value = value;
          min_idx = front_d;
        }
      }

      msg_update_shared[current_d] = min_value;
      __syncthreads();

      uint msg_offset = unary_base + current_node_h * width * n_disp + current_node_w * n_disp + current_d;
      uint msg_index_offset = tid / (n_thread_a_tree * n_trees) * height * width + current_node_h * width + current_node_w;
      uchar norm_idx = 0;

#ifdef USE_MSGNORM_NAIVE
      MsgNormNaive(param.n_disp, current_d, msg_update_shared, &norm_idx);
#else
      MsgNorm(param.n_disp, current_d, msg_update_shared, &norm_idx);
#endif

      msg_update[msg_offset] = msg_update_shared[current_d];

      if (param.is_training) {
        msg_norm_index[msg_index_offset] = norm_idx;
        msg_min_index[msg_offset] = min_idx;
      }
      __syncthreads();
    }
  }
}

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
                 at::Tensor label_all) {
  uint n_dir = msg.size(0);
  const uint batch = msg.size(1);
  const uint n_cv = msg.size(2);
  const uint height = msg.size(3);
  const uint width = msg.size(4);
  const uint n_disp = msg.size(5);
  float* unary_ptr = unary.data<float>();
  float* context_ptr = context.data<float>();
  float* edge_weight_ptr = edge_weights.data<float>();
  float* msg_ptr = msg.data<float>();  // (n_dir,batch,cv,h,w,n_disp)
  float* cost_final_ptr = cost_final.data<float>();  // (batch,cv,h,w,n_disp)
  uchar* msg_min_index_ptr = nullptr;
  uchar* msg_norm_index_ptr = nullptr;
  float* unary_update_ptr = unary_update.data<float>();
  float* msg_update_ptr = msg_update.data<float>();  // (n_dir,batch,cv,h,w,n_disp)
  uchar* label_all_ptr = nullptr;
  uint n_thread_a_tree = GetNumThreadATree(n_disp, WARP_SIZE);
  bool is_training = msg_min_index.size(0) == 0 ? false : true;
  bool is_backward = false;
  bool enable_cal_label = label_all.size(0) == 0 ? false : true;
  bool enable_sgm_single_dir = sgm_single_mode >= 0 ? true : false;

  if (enable_sgm_single_dir) n_dir = 1;

  // if (enable_sgm) {
  //   printf("Error!!! CUDA version does not support SGM but ISGMR.\n");
  //   return;
  // }

  if (is_training) {
    msg_min_index_ptr = msg_min_index.data<uchar>();  // (n_iter,n_dir,batch,cv,h,w,n_disp)
    msg_norm_index_ptr = msg_norm_index.data<uchar>();  // (n_iter,n_dir,batch,cv,h,w)
  }

  if (enable_cal_label) label_all_ptr = label_all.data<uchar>();

  // Using separate addresses for msg and index
  std::vector<float*> msg_update_address(n_dir), edge_weight_address(n_dir);
  std::vector<uchar*> msg_min_index_address(n_dir), msg_norm_index_address(n_dir);
  std::vector<Param> param_list;
  uint msg_min_size = batch * n_cv * height * width * n_disp;
  uint msg_min_index_size = n_dir * msg_min_size;
  uint msg_norm_size = msg_min_size / n_disp;
  uint msg_norm_index_size = n_dir * msg_norm_size;
  uint n_thread_unary = min(MAX_THREADS_PER_BLOCK, msg_min_size);
  uint n_block_unary = (msg_min_size + n_thread_unary - 1) / n_thread_unary;
//  cudaStream_t streams[1];
//  cudaStreamCreate(&streams[0]);

  for (uint dir = 0; dir < n_dir; ++dir) {
    msg_update_address[dir] = msg_update_ptr + dir * msg_min_size;
    edge_weight_address[dir] = edge_weight_ptr + dir * msg_norm_size;
    msg_min_index_address[dir] = nullptr;
    msg_norm_index_address[dir] = nullptr;
    uint dir_actual = enable_sgm_single_dir ? sgm_single_mode : dir;
    Param param(n_dir, batch, n_cv, height, width, n_disp, dir_actual, rho, is_backward, is_training);
    param.enable_min_a_dir = enable_min_a_dir;
    UpdateParam(&param);
    param.dir = enable_sgm_single_dir ? 0 : param.dir;
    param.dir_inv = enable_sgm_single_dir ? 1 : param.dir_inv;
    param_list.push_back(param);
  }

  // TestLoop<<<n_block_unary, n_thread_unary>>>(param_list[0], msg_min_size, msg_update_ptr);

  for (uint iter = 0; iter < n_iter; ++iter) {
    if (is_training) {
      for (uint dir = 0; dir < n_dir; ++dir) {
        msg_min_index_address[dir] = msg_min_index_ptr + iter * msg_min_index_size + dir * msg_min_size;
        msg_norm_index_address[dir] = msg_norm_index_ptr + iter * msg_norm_index_size + dir * msg_norm_size;
      }
    }

    // Horizontal
    uint n_dir_hor = enable_sgm_single_dir ? 2 : min(2, n_dir);
    for (uint idx = 0; idx < n_dir_hor; ++idx) {
      if (enable_sgm_single_dir && idx != sgm_single_mode) continue;
      uint dir = enable_sgm_single_dir ? 0 : idx;
      UpdateUnaryKernel<<<n_block_unary, n_thread_unary>>>(param_list[dir],
                                                           msg_min_size,
                                                           unary_ptr,
                                                           msg_ptr,
                                                           unary_update_ptr);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif

      uint n_threads = batch * n_cv * param_list[dir].n_trees * n_thread_a_tree;
      uint n_blocks = GetNumBlock(n_threads, n_thread_a_tree);
      HorizontalKernel<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                      n_threads,
                                                      n_thread_a_tree,
                                                      unary_update_ptr,
                                                      context_ptr,
                                                      edge_weight_address[dir],
                                                      msg_ptr,
                                                      msg_update_address[dir],
                                                      msg_min_index_address[dir],
                                                      msg_norm_index_address[dir]);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif
    }

    // Vertical
    uint n_dir_ver = enable_sgm_single_dir ? 4 : min(4, n_dir);
    for (uint idx = 2; idx < n_dir_ver; ++idx) {
      if (enable_sgm_single_dir && idx != sgm_single_mode) continue;
      uint dir = enable_sgm_single_dir ? 0 : idx;
      UpdateUnaryKernel<<<n_block_unary, n_thread_unary>>>(param_list[dir],
                                                           msg_min_size,
                                                           unary_ptr,
                                                           msg_ptr,
                                                           unary_update_ptr);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif

      uint n_threads = batch * n_cv * param_list[dir].n_trees * n_thread_a_tree;
      uint n_blocks = GetNumBlock(n_threads, n_thread_a_tree);
      DiagonalKernelWide<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                        n_threads,
                                                        n_thread_a_tree,
                                                        unary_update_ptr,
                                                        context_ptr,
                                                        edge_weight_address[dir],
                                                        msg_ptr,
                                                        msg_update_address[dir],
                                                        msg_min_index_address[dir],
                                                        msg_norm_index_address[dir]);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif
    }

    // Diagonal
    uint n_dir_dia = enable_sgm_single_dir ? 16 : min(16, n_dir);
    for (uint idx = 4; idx < n_dir_dia; ++idx) {
      if (enable_sgm_single_dir && idx != sgm_single_mode) continue;
      uint dir = enable_sgm_single_dir ? 0 : idx;
      UpdateUnaryKernel<<<n_block_unary, n_thread_unary>>>(param_list[dir],
                                                           msg_min_size,
                                                           unary_ptr,
                                                           msg_ptr,
                                                           unary_update_ptr);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif

      uint h_step_abs = std::abs(param_list[dir].h_step), w_step_abs = std::abs(param_list[dir].w_step);
      uint n_threads = batch * n_cv * param_list[dir].n_trees * n_thread_a_tree;
      uint n_blocks = GetNumBlock(n_threads, n_thread_a_tree);

      if (h_step_abs > w_step_abs)
        DiagonalKernelNarrow<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                            n_threads,
                                                            n_thread_a_tree,
                                                            unary_update_ptr,
                                                            context_ptr,
                                                            edge_weight_address[dir],
                                                            msg_ptr,
                                                            msg_update_address[dir],
                                                            msg_min_index_address[dir],
                                                            msg_norm_index_address[dir]);
      else
        DiagonalKernelWide<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                          n_threads,
                                                          n_thread_a_tree,
                                                          unary_update_ptr,
                                                          context_ptr,
                                                          edge_weight_address[dir],
                                                          msg_ptr,
                                                          msg_update_address[dir],
                                                          msg_min_index_address[dir],
                                                          msg_norm_index_address[dir]);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif
    }

    size_t copy_size = msg_min_index_size * sizeof(float);
    cudaMemcpy(msg_ptr, msg_update_ptr, copy_size, cudaMemcpyDeviceToDevice);

    if (enable_cal_label) {
      cudaMemcpy(cost_final_ptr, unary_ptr, msg_min_size * sizeof(float), cudaMemcpyDeviceToDevice);
      CostAggregateKernel<<<n_block_unary, n_thread_unary>>>(enable_sgm,
                                                             param_list[0],
                                                             msg_min_size,
                                                             msg_update_ptr,
                                                             cost_final_ptr);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif

      uint n_thread_label = min(MAX_THREADS_PER_BLOCK, msg_norm_size);
      uint n_block_label= (msg_norm_size + n_thread_label - 1) / n_thread_label;
      CalLabelKernel<<<n_block_label, n_thread_label>>>(param_list[0],
                                                        msg_norm_size,
                                                        cost_final_ptr,
                                                        label_all_ptr + iter * msg_norm_size);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif
    }
  }

  cudaMemcpy(cost_final_ptr, unary_ptr, msg_min_size * sizeof(float), cudaMemcpyDeviceToDevice);
  CostAggregateKernel<<<n_block_unary, n_thread_unary>>>(enable_sgm,
                                                         param_list[0],
                                                         msg_min_size,
                                                         msg_update_ptr,
                                                         cost_final_ptr);
#ifdef CUDA_ERROR_CHECK
  CUDAErrorCheck();
#endif

  for (uint dir = 0; dir < n_dir; ++dir) {
    if (msg_update_address[dir] != nullptr) msg_update_address[dir] = nullptr;
    if (msg_min_index_address[dir] != nullptr) msg_min_index_address[dir] = nullptr;
    if (msg_norm_index_address[dir] != nullptr) msg_norm_index_address[dir] = nullptr;
    if (edge_weight_address[dir] != nullptr) edge_weight_address[dir] = nullptr;
  }
}

// Test MsgNorm function
__global__ void MsgNormTestWrap(const uint n_disp,
                                float* value) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint current_d = tid % n_disp;
  if (current_d >= n_disp) return;

  uchar min_idx = 0;
  MsgNorm(n_disp, current_d, value, &min_idx);
  if (current_d == 0) printf("MsgNormTest value: %f, idx: %d.\n", value[0], min_idx);
}

void TestMsgNormCUDA(at::Tensor msg_norm) {
  const uint n_disp = msg_norm.size(1);
  float* value_ptr = msg_norm.data<float>();
  MsgNormTestWrap<<<1, n_disp>>>(n_disp, value_ptr);
}

// Test MultiStream function
// example from https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency
__global__ void StreamKernel(float* data) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  data[tid] = tid;
}

void TestMultiStreamCUDA(const int enable_multiple,
                          at::Tensor data) {
  const uint n_dir = data.size(0);
  const uint n_element = data.size(1);
  float* data_ptr = data.data<float>();
  int n_threads = min(MAX_THREADS_PER_BLOCK, n_element);
  int n_blocks = (n_element + n_threads - 1) / n_threads;;
  cudaStream_t streams[n_dir];

  for (uint dir = 0; dir < n_dir; ++dir) {
    if (enable_multiple == 1) {
      cudaStreamCreate(&streams[dir]);
      StreamKernel<<<n_blocks, n_threads, 0, streams[dir]>>>(data_ptr + dir * n_element);
    } else {
      StreamKernel<<<n_blocks, n_threads, 0>>>(data_ptr + dir * n_element);
    }
  }
}

#ifdef __cplusplus
  }
#endif
