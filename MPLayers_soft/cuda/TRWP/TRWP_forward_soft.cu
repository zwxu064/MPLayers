#include "TRWP.h"
#include "commonCUDA.cuh"
#include "TRWP_soft.cuh"

#ifdef __cplusplus
  extern "C" {
#endif

__device__ void DynamicProgramming(const Param param,
                                   const uint n_thread_a_tree,
                                   const int current_node_h,
                                   const int current_node_w,
                                   const int front_node_h,
                                   const int front_node_w,
                                   const float* unary_update,
                                   const float* context,
                                   const float* edge_weights,
                                   float* msg,
                                   float* msg_edge_label,
                                   uchar* msg_norm_index,
                                   float* msg_update_shared,
                                   float* msg_min_value_shared,
                                   float* msg_edge_label_shared,
                                   float* msg_edge_label_exp_shared) {
  uint height = param.height, width = param.width;
  uint n_disp = param.n_disp, n_trees = param.n_trees;
  bool is_pass_l2r = param.is_pass_l2r, is_training = param.is_training;
  float rho = param.rho;
  bool enable_seg = (n_disp == 21);

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint n_disp_with_warp = (n_disp + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
  uint max_parallel_disps = min(n_disp, blockDim.x / n_disp_with_warp);
  uint n_iters = (n_disp + max_parallel_disps - 1) / max_parallel_disps;

  uint id_base = tid / (n_trees * n_thread_a_tree);
  uint unary_base = id_base * height * width * n_disp;
  uint edge_base = id_base * height * width;
  uint msg_edge_label_base = id_base * height * width * n_disp * n_disp;
  uint offset_base = unary_base + front_node_h * width * n_disp + front_node_w * n_disp;

  float edge_weight = edge_weights[edge_base + current_node_h * width + current_node_w];

  uint current_d_base = threadIdx.x / n_disp_with_warp;
  uint front_d = threadIdx.x % n_disp_with_warp;
  uint unary_offset = offset_base + front_d;

  // Message updates, using iteration in case disp is too large that 1024 threads cannot handle n_disp*n_disp
  for (uint iter = 0; iter < n_iters; ++iter) {
    uint current_d = iter * max_parallel_disps + current_d_base;
    bool is_valid_thread = (front_d < n_disp) && (current_d < n_disp);
    bool enable_valid_assign = is_valid_thread && (threadIdx.x % n_disp_with_warp == 0);

    float context_value = 0;
    if (enable_seg)
      context_value = context[min(current_d, front_d) * n_disp + max(current_d, front_d)];
    else
      context_value = context[std::abs(int(current_d) - int(front_d))];

    uint lr_id = current_d_base * n_disp_with_warp + front_d;

    uint msg_edge_label_loc = is_pass_l2r ? (front_d * n_disp + current_d) : (current_d * n_disp + front_d);
    uint msg_edge_label_add = msg_edge_label_base + current_node_h * width * n_disp * n_disp
                              + current_node_w * n_disp * n_disp + msg_edge_label_loc;

    if (is_valid_thread) {
      float dual_value = unary_update[unary_offset] + rho * msg_update_shared[front_d] + edge_weight * context_value;
      msg_edge_label_shared[lr_id] = dual_value;
      if (is_training) msg_edge_label[msg_edge_label_add] = dual_value;
    }
    __syncthreads();

    // Find the min value among front_d
    float min_value = findMsgMin(n_disp, front_d, current_d_base, msg_edge_label_shared[lr_id]);

    if (enable_valid_assign) msg_min_value_shared[current_d] = min_value;
    __syncthreads();

    // Let msg_edge_label subtracts min_value
    if (is_valid_thread)
      msg_edge_label_exp_shared[lr_id] = __expf(-msg_edge_label_shared[lr_id] + msg_min_value_shared[current_d]);
    __syncthreads();

    // Soft message
    float sum_exp = sumMsg(n_disp, current_d_base, msg_edge_label_exp_shared[lr_id]);
    if (is_valid_thread)
      msg_edge_label_exp_shared[lr_id] = msg_edge_label_exp_shared[lr_id] * msg_edge_label_shared[lr_id] / sum_exp;
    __syncthreads();

    // Sum soft message over front_d
    float msg_soft_sum = sumMsg(n_disp, current_d_base, msg_edge_label_exp_shared[lr_id]);
    if (enable_valid_assign) msg_min_value_shared[current_d] = msg_soft_sum;
    __syncthreads();
  }

  // Norm of soft message
  uint current_d = threadIdx.x % n_disp_with_warp;
  float msg_org = msg_min_value_shared[current_d];
  uchar norm_idx = 0;
  float msg_norm = msg_org - findMsgMinIndex(n_disp, msg_min_value_shared[current_d], &norm_idx);

//      // Keep this for debugging
//      if (threadIdx.x == 0 && current_node_h == 19 && current_node_w == 47)
//        for (uint ii = 0; ii < n_disp; ++ii)
//          printf("==> %d, %f.\n", ii, msg_min_value_shared[ii]);
//       __syncthreads();

  if (threadIdx.x < n_disp) {
    uint current_d = threadIdx.x;
    uint msg_offset = unary_base + current_node_h * width * n_disp + current_node_w * n_disp + current_d;
    uint msg_index_offset = id_base * height * width + current_node_h * width + current_node_w;
    msg[msg_offset] = msg_norm;
    msg_update_shared[current_d] = msg_norm;
    if (is_training) msg_norm_index[msg_index_offset] = norm_idx;
  }
  __syncthreads();
}

__global__ void CalLabelKernelSoft(const Param param,
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

__global__ void CostAggregateKernelSoft(const Param param,
                                        const uint n_thread_required,
                                        float* msg_ptr,
                                        float* cost_final) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_thread_required) return;

  uint msg_offset = n_thread_required, n_dir = param.n_dir;
  float value = cost_final[tid];

  for (uint dir = 0; dir < n_dir; ++dir) {
   value += msg_ptr[dir * msg_offset + tid] / param.dir_weight;
  }

  cost_final[tid] = value;
  __syncthreads();
}

__global__ void UpdateUnaryKernelSoft(const Param param,
                                      int n_thread_required,
                                      float* unary,
                                      float* msg,
                                      float* unary_update) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_thread_required) return;

  uint msg_offset = n_thread_required;
  uint current_d = param.dir, dir_inv = param.dir_inv, n_dir = param.n_dir;
  float rho = param.rho, value = unary[tid];

  for (uint dir = 0; dir < n_dir; ++dir)
    value += msg[dir * msg_offset + tid];

  value -= msg[current_d * msg_offset + tid];
  value *= rho;
  value -= msg[dir_inv * msg_offset + tid];
  unary_update[tid] = value;
  __syncthreads();
}

__global__ void HorizontalKernelSoft(const Param param,
                                     const uint n_thread_required,
                                     const uint n_thread_a_tree,
                                     float* unary_update,
                                     float* context,
                                     float* edge_weights,
                                     float* msg,
                                     float* msg_edge_label,
                                     uchar* msg_norm_index) {
  // n_thread_a_tree:min(n_disp,1024/n_disp_with_warp)*n_disp_with_warp
  // msg_edge_label:(batch,cv,h,w,n_disp,n_disp)
  // msg:(batch, cv, h, w, n_disp)
  static __shared__ float msg_update_shared[MAX_DISPARITY];
  static __shared__ float msg_min_value_shared[MAX_DISPARITY];
  static __shared__ float msg_edge_label_shared[MAX_SHARED_MEM_PER_BLOCK];
  static __shared__ float msg_edge_label_exp_shared[MAX_SHARED_MEM_PER_BLOCK];
  msg_edge_label_shared[threadIdx.x] = 0;
  msg_edge_label_exp_shared[threadIdx.x] = 0;

  if (threadIdx.x < MAX_DISPARITY) {
    msg_update_shared[threadIdx.x] = 0;
    msg_min_value_shared[threadIdx.x] = 0;
  }
  __syncthreads();

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*h*n_thread_a_tree

  if (tid >= n_thread_required) return;

  uint width = param.width, n_trees = param.n_trees;
  int h_step = param.h_step, w_step = param.w_step;
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
  int h_start = tree_id, w_start = (w_step > 0) ? 0 : (width - 1);
  uint roll_step = width - 1;

  for (uint i = 0; i <= roll_step; ++i) {
    int current_node_h = h_start;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h - h_step;
    int front_node_w = current_node_w - w_step;

    if (0 <= current_node_w && current_node_w < width &&
        0 <= front_node_w && front_node_w < width)
      DynamicProgramming(param,
                         n_thread_a_tree,
                         current_node_h,
                         current_node_w,
                         front_node_h,
                         front_node_w,
                         unary_update,
                         context,
                         edge_weights,
                         msg,
                         msg_edge_label,
                         msg_norm_index,
                         msg_update_shared,
                         msg_min_value_shared,
                         msg_edge_label_shared,
                         msg_edge_label_exp_shared);
    __syncthreads();
  }
}

__global__ void DiagonalKernelNarrowSoft(const Param param,
                                         const uint n_thread_required,
                                         const uint n_thread_a_tree,
                                         float* unary_update,
                                         float* context,
                                         float* edge_weights,
                                         float* msg,
                                         float* msg_edge_label,
                                         uchar* msg_norm_index) {
  static __shared__ float msg_update_shared[MAX_DISPARITY];
  static __shared__ float msg_min_value_shared[MAX_DISPARITY];
  static __shared__ float msg_edge_label_shared[MAX_SHARED_MEM_PER_BLOCK];
  static __shared__ float msg_edge_label_exp_shared[MAX_SHARED_MEM_PER_BLOCK];
  msg_edge_label_shared[threadIdx.x] = 0;
  msg_edge_label_exp_shared[threadIdx.x] = 0;

  if (threadIdx.x < MAX_DISPARITY) {
    msg_update_shared[threadIdx.x] = 0;
    msg_min_value_shared[threadIdx.x] = 0;
  }
  __syncthreads();

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*n_trees*n_thread_a_tree

  if (tid >= n_thread_required) return;

  uint height = param.height, width = param.width, n_trees = param.n_trees;
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
  int h_step = param.h_step, w_step = param.w_step;
  uint h_step_abs = std::abs(h_step);
  int tree_id_shift = tree_id - (height - 1) * max(w_step, 0);
  int common1 = tree_id_shift % h_step_abs;
  float common2 = float(tree_id_shift) / float(h_step_abs);  // This must be float NOT int, will affect ceilf and floorf
  int h_start = 0, w_start = 0;

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
        0 <= front_node_w && front_node_w < width)
      DynamicProgramming(param,
                         n_thread_a_tree,
                         current_node_h,
                         current_node_w,
                         front_node_h,
                         front_node_w,
                         unary_update,
                         context,
                         edge_weights,
                         msg,
                         msg_edge_label,
                         msg_norm_index,
                         msg_update_shared,
                         msg_min_value_shared,
                         msg_edge_label_shared,
                         msg_edge_label_exp_shared);
    __syncthreads();
  }
}

__global__ void DiagonalKernelWideSoft(const Param param,
                                       const uint n_thread_required,
                                       const uint n_thread_a_tree,
                                       float* unary_update,
                                       float* context,
                                       float* edge_weights,
                                       float* msg,
                                       float* msg_edge_label,
                                       uchar* msg_norm_index) {
  static __shared__ float msg_update_shared[MAX_DISPARITY];
  static __shared__ float msg_min_value_shared[MAX_DISPARITY];
  static __shared__ float msg_edge_label_shared[MAX_SHARED_MEM_PER_BLOCK];
  static __shared__ float msg_edge_label_exp_shared[MAX_SHARED_MEM_PER_BLOCK];
  msg_edge_label_shared[threadIdx.x] = 0;
  msg_edge_label_exp_shared[threadIdx.x] = 0;

  if (threadIdx.x < MAX_DISPARITY) {
    msg_update_shared[threadIdx.x] = 0;
    msg_min_value_shared[threadIdx.x] = 0;
  }
  __syncthreads();

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*n_trees*n_thread_a_tree

  if (tid >= n_thread_required) return;

  uint height = param.height, width = param.width, n_trees = param.n_trees;
  int h_step = param.h_step, w_step = param.w_step;
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
  int tree_id_shift = tree_id - (height - 1) * max(w_step, 0);
  uint h_step_abs = std::abs(h_step), roll_step = (height - 1) / h_step_abs;
  int h_start = (h_step > 0) ? 0 : (height - 1), w_start = tree_id_shift;

  for (uint i = 0; i <= roll_step; ++i) {
    int current_node_h = h_start + i * h_step;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h - h_step;
    int front_node_w = current_node_w - w_step;

    if (0 <= current_node_h && current_node_h < height &&
        0 <= current_node_w && current_node_w < width &&
        0 <= front_node_h && front_node_h < height &&
        0 <= front_node_w && front_node_w < width)
      DynamicProgramming(param,
                         n_thread_a_tree,
                         current_node_h,
                         current_node_w,
                         front_node_h,
                         front_node_w,
                         unary_update,
                         context,
                         edge_weights,
                         msg,
                         msg_edge_label,
                         msg_norm_index,
                         msg_update_shared,
                         msg_min_value_shared,
                         msg_edge_label_shared,
                         msg_edge_label_exp_shared);
    __syncthreads();
  }
}

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
                     at::Tensor label_all) {
  const uint n_dir = msg.size(0);
  const uint batch = msg.size(1);
  const uint n_cv = msg.size(2);
  const uint height = msg.size(3);
  const uint width = msg.size(4);
  const uint n_disp = msg.size(5);
  float* unary_ptr = unary.data<float>();
  float* context_ptr = context.data<float>();
  float* edge_weight_ptr = edge_weights.data<float>();
  float* msg_ptr = msg.data<float>();  // (n_dir,batch,cv,h,w,n_disp)
  float* msg_edge_label_ptr = nullptr;  // (n_iter,n_dir,batch,cv,h,w,n_disp,n_disp)
  uchar* msg_norm_index_ptr = nullptr;  // (n_iter,n_dir,batch,cv,h,w)
  float* cost_final_ptr = cost_final.data<float>();  // (batch,cv,h,w,n_disp)
  float* unary_update_ptr = unary_update.data<float>();
  uchar* label_all_ptr = nullptr;
  uint n_disp_with_warp = GetNumThreadATree(n_disp, WARP_SIZE);
  uint n_thread_a_tree = min(n_disp, MAX_THREADS_PER_BLOCK / n_disp_with_warp) * n_disp_with_warp;
  bool is_training = msg_edge_label.size(0) == 0 ? false : true;
  bool enable_cal_label = label_all.size(0) == 0 ? false : true;
  bool is_backward = false;

  if (is_training) {
    msg_edge_label_ptr = msg_edge_label.data<float>();  // (n_iter,n_dir,batch,cv,h,w,n_disp,n_disp)
    msg_norm_index_ptr = msg_norm_index.data<uchar>();  // (n_iter,n_dir,batch,cv,h,w)
  }

  if (enable_cal_label) label_all_ptr = label_all.data<uchar>();

  // Using separate addresses for msg and index
  std::vector<float*> msg_addresses(n_dir), edge_weight_address(n_dir), msg_edge_label_address(n_dir);
  std::vector<uchar*> msg_norm_index_address(n_dir);
  std::vector<Param> param_list;
  uint msg_min_size = batch * n_cv * height * width * n_disp;
  uint msg_norm_size = msg_min_size / n_disp;
  uint msg_edge_label_size = n_dir * msg_min_size * n_disp;
  uint msg_norm_index_size = n_dir * msg_norm_size;
  uint n_thread_unary = min(MAX_THREADS_PER_BLOCK, msg_min_size);
  uint n_block_unary = (msg_min_size + n_thread_unary - 1) / n_thread_unary;

  for (uint dir = 0; dir < n_dir; ++dir) {
    msg_addresses[dir] = msg_ptr + dir * msg_min_size;
    edge_weight_address[dir] = edge_weight_ptr + dir * msg_norm_size;

    Param param(n_dir, batch, n_cv, height, width, n_disp, dir, rho, is_backward, is_training);
    UpdateParam(&param);
    param_list.push_back(param);
  }

  for (uint iter = 0; iter < n_iter; ++iter) {
    for (uint dir = 0; dir < n_dir; ++dir) {
      if (is_training) {
        msg_edge_label_address[dir] = msg_edge_label_ptr + iter * msg_edge_label_size + dir * msg_edge_label_size / n_dir;
        msg_norm_index_address[dir] = msg_norm_index_ptr + iter * msg_norm_index_size + dir * msg_norm_size;
      }

      UpdateUnaryKernelSoft<<<n_block_unary, n_thread_unary>>>(param_list[dir],
                                                               msg_min_size,
                                                               unary_ptr,
                                                               msg_ptr,
                                                               unary_update_ptr);
#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif

      uint n_threads = batch * n_cv * param_list[dir].n_trees * n_thread_a_tree;
      uint n_blocks = GetNumBlock(n_threads, n_thread_a_tree);

      // Horizontal
       if (dir < 2)
        HorizontalKernelSoft<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                            n_threads,
                                                            n_thread_a_tree,
                                                            unary_update_ptr,
                                                            context_ptr,
                                                            edge_weight_address[dir],
                                                            msg_addresses[dir],
                                                            msg_edge_label_address[dir],
                                                            msg_norm_index_address[dir]);

       // Vertical
       if ((2 <= dir) && (dir < 4))
         DiagonalKernelWideSoft<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                               n_threads,
                                                               n_thread_a_tree,
                                                               unary_update_ptr,
                                                               context_ptr,
                                                               edge_weight_address[dir],
                                                               msg_addresses[dir],
                                                               msg_edge_label_address[dir],
                                                               msg_norm_index_address[dir]);

       // Diagonal
       if (4 <= dir) {
         uint h_step_abs = std::abs(param_list[dir].h_step);
         uint w_step_abs = std::abs(param_list[dir].w_step);

         if (h_step_abs > w_step_abs)
           DiagonalKernelNarrowSoft<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                                   n_threads,
                                                                   n_thread_a_tree,
                                                                   unary_update_ptr,
                                                                   context_ptr,
                                                                   edge_weight_address[dir],
                                                                   msg_addresses[dir],
                                                                   msg_edge_label_address[dir],
                                                                   msg_norm_index_address[dir]);
         else
           DiagonalKernelWideSoft<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                                 n_threads,
                                                                 n_thread_a_tree,
                                                                 unary_update_ptr,
                                                                 context_ptr,
                                                                 edge_weight_address[dir],
                                                                 msg_addresses[dir],
                                                                 msg_edge_label_address[dir],
                                                                 msg_norm_index_address[dir]);
       }

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif
    }

    if (enable_cal_label) {
      cudaMemcpy(cost_final_ptr, unary_ptr, msg_min_size * sizeof(float), cudaMemcpyDeviceToDevice);
      CostAggregateKernelSoft<<<n_block_unary, n_thread_unary>>>(param_list[0],
                                                                 msg_min_size,
                                                                 msg_ptr,
                                                                 cost_final_ptr);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif

      uint n_thread_label = min(MAX_THREADS_PER_BLOCK, msg_norm_size);
      uint n_block_label= (msg_norm_size + n_thread_label - 1) / n_thread_label;
      CalLabelKernelSoft<<<n_block_label, n_thread_label>>>(param_list[0],
                                                            msg_norm_size,
                                                            cost_final_ptr,
                                                            label_all_ptr + iter * msg_norm_size);

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif
    }
  }

  cudaMemcpy(cost_final_ptr, unary_ptr, msg_min_size * sizeof(float), cudaMemcpyDeviceToDevice);
  CostAggregateKernelSoft<<<n_block_unary, n_thread_unary>>>(param_list[0],
                                                             msg_min_size,
                                                             msg_ptr,
                                                             cost_final_ptr);

#ifdef CUDA_ERROR_CHECK
  CUDAErrorCheck();
#endif

  for (uint dir = 0; dir < n_dir; ++dir) {
    if (msg_addresses[dir] != nullptr) msg_addresses[dir] = nullptr;
    if (msg_edge_label_address[dir] != nullptr) msg_edge_label_address[dir] = nullptr;
    if (msg_norm_index_address[dir] != nullptr) msg_norm_index_address[dir] = nullptr;
    if (edge_weight_address[dir] != nullptr) edge_weight_address[dir] = nullptr;
  }
}

#ifdef __cplusplus
  }
#endif
