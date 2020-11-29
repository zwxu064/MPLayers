#include "TRWP.h"
#include "commonCUDA.cuh"

#ifdef __cplusplus
  extern "C" {
#endif

__global__ void CostAggregateKernelBack(const Param param,
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

__global__ void UpdateUnaryKernelBack(const Param param,
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

__global__ void HorizontalKernelBack(const Param param,
                                     const uint n_thread_required,
                                     const uint n_thread_a_tree,
                                     const float* context,
                                     const float* edge_weights,
                                     const uchar* msg_min_index,
                                     const uchar* msg_norm_index,
                                     float* dmsg,
                                     float* dunary_update,
                                     float* dcontext,
                                     float* dedge_weights) {
  static __shared__ float dmsg_update_shared[MAX_DISPARITY];

  uint height = param.height, width = param.width;
  uint n_disp = param.n_disp, n_trees = param.n_trees;
  float rho = param.rho;
  int w_step = param.w_step;
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;  // batch*cv*h*n_thread_a_tree
  uint current_d = threadIdx.x % n_thread_a_tree;
  bool enable_seg = (n_disp == 21);

  if (tid >= n_thread_required) return;
  if (current_d >= n_disp) return;

  uint unary_base = tid / (n_trees * n_thread_a_tree) * height * width * n_disp;
  uint tree_id = (tid / n_thread_a_tree) % n_trees;
  int h_start = tree_id, w_start = (w_step > 0) ? 0 : (width - 1);
  uint edge_base = tid / (n_thread_a_tree * n_trees) * height * width;

  // The front node is in accordance with forward pass, use + *_step
  // msg_min_index(batch,n_cv,h,w,n_disp)
  for (uint i = 0; i < width; ++i) {
    int current_node_h = h_start;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h;
    int front_node_w = current_node_w + w_step;

    if (0 <= current_node_w && current_node_w < width &&
        0 <= front_node_w && front_node_w < width) {
      uint msg_offset_base = unary_base + current_node_h * width * n_disp + current_node_w * n_disp;
      uint msg_offset = msg_offset_base + current_d;
      uint msg_index_offset = tid / n_thread_a_tree * width + current_node_w;
      uint edge_weight_offset = edge_base + current_node_h * width + current_node_w;

      dmsg_update_shared[current_d] = dmsg[msg_offset];
      __syncthreads();
      MsgNormNaiveBack(param.n_disp, current_d, msg_norm_index[msg_index_offset], dmsg_update_shared);

      uint front_d = uint(msg_min_index[msg_offset]);
      float value = dmsg_update_shared[current_d];
      float value_rho = rho * value;
      uint offset = unary_base + front_node_h * width * n_disp + front_node_w * n_disp + front_d;
      uint context_offset = 0;
      if (enable_seg)
        context_offset = min(current_d, front_d) * n_disp + max(current_d, front_d);
      else
        context_offset = std::abs(int(current_d) - int(front_d));

      atomicAdd(&dunary_update[offset], value);
      atomicAdd(&dmsg[offset], value_rho);
      atomicAdd(&dedge_weights[edge_weight_offset], context[context_offset] * value);
      atomicAdd(&dcontext[context_offset], edge_weights[edge_weight_offset] * value);

      __syncthreads();
    }
  }
}

__global__ void DiagonalKernelNarrowBack(const Param param,
                                         const uint n_thread_required,
                                         const uint n_thread_a_tree,
                                         const float* context,
                                         const float* edge_weights,
                                         const uchar* msg_min_index,
                                         const uchar* msg_norm_index,
                                         float* dmsg,
                                         float* dunary_update,
                                         float* dcontext,
                                         float* dedge_weights) {
  static __shared__ float dmsg_update_shared[MAX_DISPARITY];

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
    int front_node_h = current_node_h + h_step;
    int front_node_w = current_node_w + w_step;

    if (0 <= current_node_h && current_node_h < height &&
        0 <= current_node_w && current_node_w < width &&
        0 <= front_node_h && front_node_h < height &&
        0 <= front_node_w && front_node_w < width) {
      uint msg_offset_base = unary_base + current_node_h * width * n_disp + current_node_w * n_disp;
      uint msg_offset = msg_offset_base + current_d;
      uint msg_index_offset = tid / (n_thread_a_tree * n_trees) * height * width + current_node_h * width + current_node_w;
      uint edge_weight_offset = edge_base + current_node_h * width + current_node_w;

      dmsg_update_shared[current_d] = dmsg[msg_offset];
      __syncthreads();
      MsgNormNaiveBack(param.n_disp, current_d, msg_norm_index[msg_index_offset], dmsg_update_shared);

      uint front_d = uint(msg_min_index[msg_offset]);
      float value = dmsg_update_shared[current_d];
      float value_rho = rho * value;
      uint offset = unary_base + front_node_h * width * n_disp + front_node_w * n_disp + front_d;
      uint context_offset = 0;
      if (enable_seg)
        context_offset = min(current_d, front_d) * n_disp + max(current_d, front_d);
      else
        context_offset = std::abs(int(current_d) - int(front_d));

      atomicAdd(&dunary_update[offset], value);
      atomicAdd(&dmsg[offset], value_rho);
      atomicAdd(&dedge_weights[edge_weight_offset], context[context_offset] * value);
      atomicAdd(&dcontext[context_offset], edge_weights[edge_weight_offset] * value);

      __syncthreads();
    }
  }
}

__global__ void DiagonalKernelWideBack(const Param param,
                                       const uint n_thread_required,
                                       const uint n_thread_a_tree,
                                       const float* context,
                                       const float* edge_weights,
                                       const uchar* msg_min_index,
                                       const uchar* msg_norm_index,
                                       float* dmsg,
                                       float* dunary_update,
                                       float* dcontext,
                                       float* dedge_weights) {
  static __shared__ float dmsg_update_shared[MAX_DISPARITY];

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
  int tree_id_shift = tree_id - (height - 1) * max(w_step, 0);
  uint h_step_abs = std::abs(h_step), roll_step = (height - 1) / h_step_abs;
  int h_start = (h_step > 0) ? 0 : (height - 1), w_start = tree_id_shift;
  uint edge_base = tid / (n_thread_a_tree * n_trees) * height * width;

  for (uint i = 0; i <= roll_step; ++i) {
    int current_node_h = h_start + i * h_step;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h + h_step;
    int front_node_w = current_node_w + w_step;

    if (0 <= current_node_h && current_node_h < height &&
        0 <= current_node_w && current_node_w < width &&
        0 <= front_node_h && front_node_h < height &&
        0 <= front_node_w && front_node_w < width) {
      uint msg_offset_base = unary_base + current_node_h * width * n_disp + current_node_w * n_disp;
      uint msg_offset = msg_offset_base + current_d;
      uint msg_index_offset = tid / (n_thread_a_tree * n_trees) * height * width + current_node_h * width + current_node_w;
      uint edge_weight_offset = edge_base + current_node_h * width + current_node_w;

      dmsg_update_shared[current_d] = dmsg[msg_offset];
      __syncthreads();
      MsgNormNaiveBack(param.n_disp, current_d, msg_norm_index[msg_index_offset], dmsg_update_shared);

      uint front_d = uint(msg_min_index[msg_offset]);
      float value = dmsg_update_shared[current_d];
      float value_rho = rho * value;
      uint offset = unary_base + front_node_h * width * n_disp + front_node_w * n_disp + front_d;
      uint context_offset = 0;
      if (enable_seg)
        context_offset = min(current_d, front_d) * n_disp + max(current_d, front_d);
      else
        context_offset = std::abs(int(current_d) - int(front_d));

      atomicAdd(&dunary_update[offset], value);
      atomicAdd(&dmsg[offset], value_rho);
      atomicAdd(&dedge_weights[edge_weight_offset], context[context_offset] * value);
      atomicAdd(&dcontext[context_offset], edge_weights[edge_weight_offset] * value);

      __syncthreads();
    }
  }
}

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
                  at::Tensor dunary_update) {
  const uint n_iter = msg_min_index.size(0);
  const uint n_dir = msg_min_index.size(1);
  const uint batch = msg_min_index.size(2);
  const uint n_cv = msg_min_index.size(3);
  const uint height = msg_min_index.size(4);
  const uint width = msg_min_index.size(5);
  const uint n_disp = msg_min_index.size(6);
  float* context_ptr = context.data<float>();
  float* edge_weight_ptr = edge_weights.data<float>();
  float* dcost_final_ptr = dcost_final.data<float>();
  uchar* msg_min_index_ptr = msg_min_index.data<uchar>();  // (n_iter,n_dir,batch,n_cv,h,w,n_disp)
  uchar* msg_norm_index_ptr = msg_norm_index.data<uchar>();  // (n_iter,n_dir,batch,n_cv,h,w)
  float* dunary_ptr = dunary.data<float>();  // (batch,n_cv,h,w,n_disp)
  float* dcontext_ptr = dcontext.data<float>();
  float* dedge_weight_ptr = dedge_weights.data<float>();
  float* dmsg_ptr = dmsg.data<float>();
  float* dunary_update_ptr = dunary_update.data<float>();
  uint n_thread_a_tree = GetNumThreadATree(n_disp, WARP_SIZE);
  bool is_backward = true, is_training = true;

  std::vector<float*> dmsg_address(n_dir), edge_weight_address(n_dir), dedge_weight_address(n_dir);
  std::vector<uchar*> msg_min_index_address(n_dir), msg_norm_index_address(n_dir);
  std::vector<Param> param_list;
  uint msg_min_size = batch * n_cv * height * width * n_disp;
  uint msg_min_index_size = n_dir * msg_min_size;
  uint msg_norm_size = msg_min_size / n_disp;
  uint msg_norm_index_size = n_dir * msg_norm_size;
  uint n_thread_unary = min(MAX_THREADS_PER_BLOCK, msg_min_size);
  uint n_block_unary = (msg_min_size + n_thread_unary - 1) / n_thread_unary;
  uint n_thread_msg_norm = min(MAX_THREADS_PER_BLOCK, msg_norm_size);

  for (int dir = 0; dir < n_dir; ++dir) {
    edge_weight_address[dir] = edge_weight_ptr + dir * msg_norm_size;
    dedge_weight_address[dir] = dedge_weight_ptr + dir * msg_norm_size;
    dmsg_address[dir] = dmsg_ptr + dir * msg_min_size;
    Param param(n_dir, batch, n_cv, height, width, n_disp, dir, rho, is_backward, is_training);
    UpdateParam(&param);
    param_list.push_back(param);
  }

  CostAggregateKernelBack<<<n_block_unary, n_thread_unary>>>(param_list[0],
                                                             msg_min_size,
                                                             dcost_final_ptr,
                                                             dunary_ptr,
                                                             dmsg_ptr);
#ifdef CUDA_ERROR_CHECK
  CUDAErrorCheck();
#endif

  for (int iter = n_iter - 1; iter >= 0; --iter) {
    for (int dir = n_dir - 1; dir >= 0; --dir) {
      msg_min_index_address[dir] = msg_min_index_ptr + iter * msg_min_index_size + dir * msg_min_size;
      msg_norm_index_address[dir] = msg_norm_index_ptr + iter * msg_norm_index_size + dir * msg_norm_size;

      uint n_threads = batch * n_cv * param_list[dir].n_trees * n_thread_a_tree;
      uint n_blocks = GetNumBlock(n_threads, n_thread_a_tree);

      // Diagonal
      if (4 <= dir) {
        uint h_step_abs = std::abs(param_list[dir].h_step);
        uint w_step_abs = std::abs(param_list[dir].w_step);

        if (h_step_abs > w_step_abs) {
          DiagonalKernelNarrowBack<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                                  n_threads,
                                                                  n_thread_a_tree,
                                                                  context_ptr,
                                                                  edge_weight_address[dir],
                                                                  msg_min_index_address[dir],
                                                                  msg_norm_index_address[dir],
                                                                  dmsg_address[dir],
                                                                  dunary_update_ptr,
                                                                  dcontext_ptr,
                                                                  dedge_weight_address[dir]);
        } else {
          DiagonalKernelWideBack<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                                n_threads,
                                                                n_thread_a_tree,
                                                                context_ptr,
                                                                edge_weight_address[dir],
                                                                msg_min_index_address[dir],
                                                                msg_norm_index_address[dir],
                                                                dmsg_address[dir],
                                                                dunary_update_ptr,
                                                                dcontext_ptr,
                                                                dedge_weight_address[dir]);
        }
      }

      // Vertical
      if ((2 <= dir) && (dir < 4)) {
        DiagonalKernelWideBack<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                              n_threads,
                                                              n_thread_a_tree,
                                                              context_ptr,
                                                              edge_weight_address[dir],
                                                              msg_min_index_address[dir],
                                                              msg_norm_index_address[dir],
                                                              dmsg_address[dir],
                                                              dunary_update_ptr,
                                                              dcontext_ptr,
                                                              dedge_weight_address[dir]);
      }

      // Horizontal
      if (dir < 2) {
        HorizontalKernelBack<<<n_blocks, n_thread_a_tree>>>(param_list[dir],
                                                            n_threads,
                                                            n_thread_a_tree,
                                                            context_ptr,
                                                            edge_weight_address[dir],
                                                            msg_min_index_address[dir],
                                                            msg_norm_index_address[dir],
                                                            dmsg_address[dir],
                                                            dunary_update_ptr,
                                                            dcontext_ptr,
                                                            dedge_weight_address[dir]);
      }

#ifdef CUDA_ERROR_CHECK
      CUDAErrorCheck();
#endif

      UpdateUnaryKernelBack<<<n_block_unary, n_thread_unary>>>(param_list[dir],
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
    if (msg_min_index_address[dir] != nullptr) msg_min_index_address[dir] = nullptr;
    if (msg_norm_index_address[dir] != nullptr) msg_norm_index_address[dir] = nullptr;
    if (edge_weight_address[dir] != nullptr) edge_weight_address[dir] = nullptr;
    if (dedge_weight_address[dir] != nullptr) dedge_weight_address[dir] = nullptr;
  }
}

#ifdef __cplusplus
  }
#endif
