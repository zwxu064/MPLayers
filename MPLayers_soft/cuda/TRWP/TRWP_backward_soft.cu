#include "TRWP.h"
#include "commonCUDA.cuh"
#include "TRWP_soft.cuh"

#ifdef __cplusplus
  extern "C" {
#endif

__device__ void DynamicProgrammingBack(const Param param,
                                       const uint n_thread_a_tree,
                                       const uint current_node_h,
                                       const uint current_node_w,
                                       const uint front_node_h,
                                       const uint front_node_w,
                                       const float* context,
                                       const float* edge_weights,
                                       const float* msg_edge_label,
                                       const uchar* msg_norm_index,
                                       float* dmsg,
                                       float* dunary_update,
                                       float* dcontext,
                                       float* dedge_weights,
                                       float* dmsg_update_shared,
                                       float* msg_min_value_shared,
                                       float* msg_edge_label_shared,
                                       float* msg_edge_label_exp_shared) {
  uint height = param.height, width = param.width;
  uint n_disp = param.n_disp, n_trees = param.n_trees;
  bool is_pass_l2r = param.is_pass_l2r;
  float rho = param.rho;

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint n_disp_with_warp = (n_disp + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
  uint max_parallel_disps = min(n_disp, blockDim.x / n_disp_with_warp);
  uint n_iters = (n_disp + max_parallel_disps - 1) / max_parallel_disps;
  bool enable_seg = (n_disp == 21);

  uint id_base = tid / (n_trees * n_thread_a_tree);
  uint unary_base = id_base * height * width * n_disp;
  uint edge_base = id_base * height * width;
  uint msg_edge_label_base = id_base * height * width * n_disp * n_disp;
  uint current_d_base = threadIdx.x / n_disp_with_warp;
  uint msg_index_offset = id_base * height * width + current_node_h * width + current_node_w;
  uchar norm_index = msg_norm_index[msg_index_offset];

  if (threadIdx.x < n_disp) {
    uint current_d = threadIdx.x;
    uint msg_offset = unary_base + current_node_h * width * n_disp + current_node_w * n_disp + current_d;
    dmsg_update_shared[threadIdx.x] = dmsg[msg_offset];
  }
  __syncthreads();

  // Back norm
  uint current_d_4norm = threadIdx.x % n_disp_with_warp;
  float gradient = 0;

  // A patch: current_d_4norm above may exceed MAX_DISPARITY
  if (current_d_4norm < MAX_DISPARITY) gradient = dmsg_update_shared[current_d_4norm];
  __syncthreads();

  float gradient_sum = sumMsg(n_disp, current_d_base, gradient);
  if (threadIdx.x == 0) dmsg_update_shared[norm_index] -= gradient_sum;
  __syncthreads();

  uint offset_base = unary_base + front_node_h * width * n_disp + front_node_w * n_disp;
  uint front_d = threadIdx.x % n_disp_with_warp;
  uint unary_offset = offset_base + front_d;

  for (uint iter = 0; iter < n_iters; ++iter) {
    uint current_d = iter * max_parallel_disps + current_d_base;
    bool is_valid_thread = (front_d < n_disp) && (current_d < n_disp);
    bool enable_valid_assign = is_valid_thread && (threadIdx.x % n_disp_with_warp == 0);
    uint lr_id = current_d_base * n_disp_with_warp + front_d;
    uint msg_edge_label_loc = is_pass_l2r ? (front_d * n_disp + current_d) : (current_d * n_disp + front_d);
    uint msg_edge_label_add = msg_edge_label_base + current_node_h * width * n_disp * n_disp
                              + current_node_w * n_disp * n_disp + msg_edge_label_loc;

    // Calculate p * (1 - msg_edge_label + msg)
    if (is_valid_thread) msg_edge_label_shared[lr_id] = msg_edge_label[msg_edge_label_add];
    __syncthreads();

    // ==== BEGIN: from forward, re-calculate prob and msg_soft_sum
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
    float prob = msg_edge_label_exp_shared[lr_id] / sum_exp;
    float msg_soft = prob * msg_edge_label_shared[lr_id];

    if (is_valid_thread) msg_edge_label_exp_shared[lr_id] = msg_soft;
    __syncthreads();

    // Sum soft message over front_d
    float msg_soft_sum = sumMsg(n_disp, current_d_base, msg_edge_label_exp_shared[lr_id]);
    if (enable_valid_assign) msg_min_value_shared[current_d] = msg_soft_sum;
    __syncthreads();
    // ==== END: From forward

    if (is_valid_thread) {
      // Calculate dmsg_edge_label
      float dmsg_sum = dmsg_update_shared[current_d];
      float msg_edge_label_one = msg_edge_label_shared[lr_id];
      float dmsg_edge_label = dmsg_sum * prob * (1 - msg_edge_label_one + msg_soft_sum);

      uint context_loc = 0;
      if (enable_seg)
        context_loc = min(current_d, front_d) * n_disp + max(current_d, front_d);
      else
        context_loc = std::abs(int(current_d) - int(front_d));

      uint edge_weight_loc = edge_base + current_node_h * width + current_node_w;

      atomicAdd(&dunary_update[unary_offset], dmsg_edge_label);
      atomicAdd(&dmsg[unary_offset], rho * dmsg_edge_label);
      atomicAdd(&dedge_weights[edge_weight_loc], context[context_loc] * dmsg_edge_label);
      atomicAdd(&dcontext[context_loc], edge_weights[edge_weight_loc] * dmsg_edge_label);
    }
    __syncthreads();
  }
}

__global__ void CostAggregateKernelSoftBack(const Param param,
                                            const uint n_thread_required,
                                            float* dcost_final_ptr,
                                            float* dunary,
                                            float* dmsg_ptr) {
  // cost_final=unary+sum{msg_update}
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_thread_required) return;

  float dcost_final_value = dcost_final_ptr[tid];
  dunary[tid] = dcost_final_value;

  for (uint dir = 0; dir < param.n_dir; ++dir)
    dmsg_ptr[dir * n_thread_required + tid] = dcost_final_value;

  __syncthreads();
}

__global__ void UpdateUnaryKernelSoftBack(const Param param,
                                          const uint n_thread_required,
                                          float* dunary_update_ptr,
                                          float* dunary_ptr,
                                          float* dmsg_ptr) {
  // unary_update=rho*(unary+sum{msg}-msg_dir)-msg_dir_inv
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*n_cv*h*w*n_disp
  if (tid >= n_thread_required) return;

  uint dir = param.dir, dir_inv = param.dir_inv, n_dir = param.n_dir;
  float rho = param.rho;
  float dunary_update_value = dunary_update_ptr[tid];
  float dunary_update_value_rho = rho * dunary_update_value;

  for (uint dir = 0; dir < n_dir; ++dir)
    atomicAdd(&dmsg_ptr[dir * n_thread_required + tid], dunary_update_value_rho);

  atomicAdd(&dunary_ptr[tid], dunary_update_value_rho);
  atomicAdd(&dmsg_ptr[dir * n_thread_required + tid], -dunary_update_value_rho);
  atomicAdd(&dmsg_ptr[dir_inv * n_thread_required + tid], -dunary_update_value);
  __syncthreads();
}

__global__ void HorizontalKernelSoftBack(const Param param,
                                         const uint n_thread_required,
                                         const uint n_thread_a_tree,
                                         const float* context,
                                         const float* edge_weights,
                                         const float* msg_edge_label,
                                         const uchar* msg_norm_index,
                                         float* dmsg,
                                         float* dunary_update,
                                         float* dcontext,
                                         float* dedge_weights) {
  static __shared__ float dmsg_update_shared[MAX_DISPARITY];
  static __shared__ float msg_min_value_shared[MAX_DISPARITY];
  static __shared__ float msg_edge_label_shared[MAX_SHARED_MEM_PER_BLOCK];
  static __shared__ float msg_edge_label_exp_shared[MAX_SHARED_MEM_PER_BLOCK];

  msg_edge_label_shared[threadIdx.x] = 0;
  msg_edge_label_exp_shared[threadIdx.x] = 0;

  if (threadIdx.x < MAX_DISPARITY) {
    msg_min_value_shared[threadIdx.x] = 0;
    dmsg_update_shared[threadIdx.x] = 0;
  }
  __syncthreads();

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*h*n_thread_a_tree

  if (tid >= n_thread_required) return;

  uint width = param.width, n_trees = param.n_trees;
  int w_step = param.w_step;
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
  int h_start = tree_id, w_start = (w_step > 0) ? 0 : (width - 1);
  uint roll_step = width - 1;

  // The front node is in accordance with forward pass, use + *_step
  // msg_min_index(batch,n_cv,h,w,n_disp)
  for (uint i = 0; i <= roll_step; ++i) {
    int current_node_h = h_start;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h;
    int front_node_w = current_node_w + w_step;

    if (0 <= current_node_w && current_node_w < width &&
        0 <= front_node_w && front_node_w < width)
      DynamicProgrammingBack(param,
                             n_thread_a_tree,
                             current_node_h,
                             current_node_w,
                             front_node_h,
                             front_node_w,
                             context,
                             edge_weights,
                             msg_edge_label,
                             msg_norm_index,
                             dmsg,
                             dunary_update,
                             dcontext,
                             dedge_weights,
                             dmsg_update_shared,
                             msg_min_value_shared,
                             msg_edge_label_shared,
                             msg_edge_label_exp_shared);
    __syncthreads();
  }
}

__global__ void DiagonalKernelNarrowSoftBack(const Param param,
                                             const uint n_thread_required,
                                             const uint n_thread_a_tree,
                                             const float* context,
                                             const float* edge_weights,
                                             const float* msg_edge_label,
                                             const uchar* msg_norm_index,
                                             float* dmsg,
                                             float* dunary_update,
                                             float* dcontext,
                                             float* dedge_weights) {
  static __shared__ float dmsg_update_shared[MAX_DISPARITY];
  static __shared__ float msg_min_value_shared[MAX_DISPARITY];
  static __shared__ float msg_edge_label_shared[MAX_SHARED_MEM_PER_BLOCK];
  static __shared__ float msg_edge_label_exp_shared[MAX_SHARED_MEM_PER_BLOCK];

  msg_edge_label_shared[threadIdx.x] = 0;
  msg_edge_label_exp_shared[threadIdx.x] = 0;

  if (threadIdx.x < MAX_DISPARITY) {
    msg_min_value_shared[threadIdx.x] = 0;
    dmsg_update_shared[threadIdx.x] = 0;
  }
  __syncthreads();

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*h*n_thread_a_tree

  if (tid >= n_thread_required) return;

  uint height = param.height, width = param.width, n_trees = param.n_trees;
  int h_step = param.h_step, w_step = param.w_step;
  uint h_step_abs = std::abs(h_step);
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
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

  // The front node is in accordance with forward pass, use + *_step
  // msg_min_index(batch,n_cv,h,w,n_disp)
  for (uint i = 0; i <= roll_step; ++i) {
    int current_node_h = h_start + i * h_step;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h + h_step;
    int front_node_w = current_node_w + w_step;

    if (0 <= current_node_h && current_node_h < height &&
        0 <= current_node_w && current_node_w < width &&
        0 <= front_node_h && front_node_h < height &&
        0 <= front_node_w && front_node_w < width)
      DynamicProgrammingBack(param,
                             n_thread_a_tree,
                             current_node_h,
                             current_node_w,
                             front_node_h,
                             front_node_w,
                             context,
                             edge_weights,
                             msg_edge_label,
                             msg_norm_index,
                             dmsg,
                             dunary_update,
                             dcontext,
                             dedge_weights,
                             dmsg_update_shared,
                             msg_min_value_shared,
                             msg_edge_label_shared,
                             msg_edge_label_exp_shared);
    __syncthreads();
  }
}

__global__ void DiagonalKernelWideSoftBack(const Param param,
                                           const uint n_thread_required,
                                           const uint n_thread_a_tree,
                                           const float* context,
                                           const float* edge_weights,
                                           const float* msg_edge_label,
                                           const uchar* msg_norm_index,
                                           float* dmsg,
                                           float* dunary_update,
                                           float* dcontext,
                                           float* dedge_weights) {
  static __shared__ float dmsg_update_shared[MAX_DISPARITY];
  static __shared__ float msg_min_value_shared[MAX_DISPARITY];
  static __shared__ float msg_edge_label_shared[MAX_SHARED_MEM_PER_BLOCK];
  static __shared__ float msg_edge_label_exp_shared[MAX_SHARED_MEM_PER_BLOCK];

  msg_edge_label_shared[threadIdx.x] = 0;
  msg_edge_label_exp_shared[threadIdx.x] = 0;

  if (threadIdx.x < MAX_DISPARITY) {
    msg_min_value_shared[threadIdx.x] = 0;
    dmsg_update_shared[threadIdx.x] = 0;
  }
  __syncthreads();

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*h*n_thread_a_tree

  if (tid >= n_thread_required) return;

  uint height = param.height, width = param.width, n_trees = param.n_trees;
  int h_step = param.h_step, w_step = param.w_step;
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
  int tree_id_shift = tree_id - (height - 1) * max(w_step, 0);
  uint h_step_abs = std::abs(h_step), roll_step = (height - 1) / h_step_abs;
  int h_start = (h_step > 0) ? 0 : (height - 1), w_start = tree_id_shift;

  // The front node is in accordance with forward pass, use + *_step
  // msg_min_index(batch,n_cv,h,w,n_disp)
  for (uint i = 0; i <= roll_step; ++i) {
    int current_node_h = h_start + i * h_step;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h + h_step;
    int front_node_w = current_node_w + w_step;

    if (0 <= current_node_h && current_node_h < height &&
        0 <= current_node_w && current_node_w < width &&
        0 <= front_node_h && front_node_h < height &&
        0 <= front_node_w && front_node_w < width)
      DynamicProgrammingBack(param,
                             n_thread_a_tree,
                             current_node_h,
                             current_node_w,
                             front_node_h,
                             front_node_w,
                             context,
                             edge_weights,
                             msg_edge_label,
                             msg_norm_index,
                             dmsg,
                             dunary_update,
                             dcontext,
                             dedge_weights,
                             dmsg_update_shared,
                             msg_min_value_shared,
                             msg_edge_label_shared,
                             msg_edge_label_exp_shared);
    __syncthreads();
  }
}

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
                      at::Tensor dunary_update) {
  const uint n_iter = msg_edge_label.size(0);
  const uint n_dir = msg_edge_label.size(1);
  const uint batch = msg_edge_label.size(2);
  const uint n_cv = msg_edge_label.size(3);
  const uint height = msg_edge_label.size(4);
  const uint width = msg_edge_label.size(5);
  const uint n_disp = msg_edge_label.size(6);
  float* dcost_final_ptr = dcost_final.data<float>();
  float* context_ptr = context.data<float>();
  float* edge_weight_ptr = edge_weights.data<float>();
  float* msg_edge_label_ptr = msg_edge_label.data<float>();  // (n_iter,n_dir,batch,n_cv,h,w,n_disp,n_disp)
  uchar* msg_norm_index_ptr = msg_norm_index.data<uchar>();  // (n_iter,n_dir,batch,n_cv,h,w)
  float* dunary_ptr = dunary.data<float>();  // (batch,n_cv,h,w,n_disp)
  float* dcontext_ptr = dcontext.data<float>();
  float* dedge_weight_ptr = dedge_weights.data<float>();
  float* dmsg_ptr = dmsg.data<float>();
  float* dunary_update_ptr = dunary_update.data<float>();
  uint n_disp_with_warp = GetNumThreadATree(n_disp, WARP_SIZE);
  uint n_thread_a_tree = min(n_disp, MAX_THREADS_PER_BLOCK / n_disp_with_warp) * n_disp_with_warp;
  bool is_backward = true, is_training = true;

  std::vector<float*> dmsg_address(n_dir), edge_weight_address(n_dir);
  std::vector<float*> dedge_weight_address(n_dir), msg_edge_label_address(n_dir);
  std::vector<uchar*> msg_norm_index_address(n_dir);
  std::vector<Param> param_list;
  uint msg_min_size = batch * n_cv * height * width * n_disp;
  uint msg_norm_size = msg_min_size / n_disp;
  uint msg_edge_label_size = n_dir * msg_min_size * n_disp;
  uint msg_norm_index_size = n_dir * msg_norm_size;
  uint n_thread_unary = min(MAX_THREADS_PER_BLOCK, msg_min_size);
  uint n_block_unary = (msg_min_size + n_thread_unary - 1) / n_thread_unary;

   for (int dir = 0; dir < n_dir; ++dir) {
     edge_weight_address[dir] = edge_weight_ptr + dir * msg_norm_size;
     dedge_weight_address[dir] = dedge_weight_ptr + dir * msg_norm_size;
     dmsg_address[dir] = dmsg_ptr + dir * msg_min_size;
     Param param(n_dir, batch, n_cv, height, width, n_disp, dir, rho, is_backward, is_training);
     UpdateParam(&param);
     param_list.push_back(param);
   }

   CostAggregateKernelSoftBack<<<n_block_unary, n_thread_unary>>>(param_list[0],
                                                                  msg_min_size,
                                                                  dcost_final_ptr,
                                                                  dunary_ptr,
                                                                  dmsg_ptr);
 #ifdef CUDA_ERROR_CHECK
   CUDAErrorCheck();
 #endif

   for (int iter = n_iter - 1; iter >= 0; --iter) {
     for (int dir = n_dir - 1; dir >= 0; --dir) {
       msg_edge_label_address[dir] = msg_edge_label_ptr + iter * msg_edge_label_size + dir * msg_edge_label_size / n_dir;
       msg_norm_index_address[dir] = msg_norm_index_ptr + iter * msg_norm_index_size + dir * msg_norm_size;

       uint n_threads = batch * n_cv * param_list[dir].n_trees * n_thread_a_tree;
       uint n_blocks = GetNumBlock(n_threads, n_thread_a_tree);

       // Diagonal
       if (4 <= dir) {
         uint h_step_abs = std::abs(param_list[dir].h_step);
         uint w_step_abs = std::abs(param_list[dir].w_step);

         if (h_step_abs > w_step_abs)
           DiagonalKernelNarrowSoftBack<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                                       n_threads,
                                                                       n_thread_a_tree,
                                                                       context_ptr,
                                                                       edge_weight_address[dir],
                                                                       msg_edge_label_address[dir],
                                                                       msg_norm_index_address[dir],
                                                                       dmsg_address[dir],
                                                                       dunary_update_ptr,
                                                                       dcontext_ptr,
                                                                       dedge_weight_address[dir]);
         else
           DiagonalKernelWideSoftBack<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                                     n_threads,
                                                                     n_thread_a_tree,
                                                                     context_ptr,
                                                                     edge_weight_address[dir],
                                                                     msg_edge_label_address[dir],
                                                                     msg_norm_index_address[dir],
                                                                     dmsg_address[dir],
                                                                     dunary_update_ptr,
                                                                     dcontext_ptr,
                                                                     dedge_weight_address[dir]);
       }

       // Vertical
       if ((2 <= dir) && (dir < 4))
         DiagonalKernelWideSoftBack<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                                   n_threads,
                                                                   n_thread_a_tree,
                                                                   context_ptr,
                                                                   edge_weight_address[dir],
                                                                   msg_edge_label_address[dir],
                                                                   msg_norm_index_address[dir],
                                                                   dmsg_address[dir],
                                                                   dunary_update_ptr,
                                                                   dcontext_ptr,
                                                                   dedge_weight_address[dir]);

       // Horizontal
       if (dir < 2)
         HorizontalKernelSoftBack<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                                 n_threads,
                                                                 n_thread_a_tree,
                                                                 context_ptr,
                                                                 edge_weight_address[dir],
                                                                 msg_edge_label_address[dir],
                                                                 msg_norm_index_address[dir],
                                                                 dmsg_address[dir],
                                                                 dunary_update_ptr,
                                                                 dcontext_ptr,
                                                                 dedge_weight_address[dir]);

 #ifdef CUDA_ERROR_CHECK
       CUDAErrorCheck();
 #endif

       UpdateUnaryKernelSoftBack<<<n_block_unary, n_thread_unary>>>(param_list[dir],
                                                                    msg_min_size,
                                                                    dunary_update_ptr,
                                                                    dunary_ptr,
                                                                    dmsg_ptr);

 #ifdef CUDA_ERROR_CHECK
       CUDAErrorCheck();
 #endif

       cudaMemset(dunary_update_ptr, 0, msg_min_size * sizeof(float));
       cudaMemset(dmsg_address[dir], 0, msg_min_size * sizeof(float));
     }
   }

   for (uint dir = 0; dir < n_dir; ++dir) {
     if (dmsg_address[dir] != nullptr) dmsg_address[dir] = nullptr;
     if (msg_edge_label_address[dir] != nullptr) msg_edge_label_address[dir] = nullptr;
     if (msg_norm_index_address[dir] != nullptr) msg_norm_index_address[dir] = nullptr;
     if (edge_weight_address[dir] != nullptr) edge_weight_address[dir] = nullptr;
     if (dedge_weight_address[dir] != nullptr) dedge_weight_address[dir] = nullptr;
   }
}

#ifdef __cplusplus
  }
#endif
