#include <pybind11/pybind11.h>
#include "TRWP.h"

namespace py = pybind11;

#ifdef __cplusplus
  extern "C" {
#endif

inline void DynamicProgramming(const Param param,
                               const uint bv_offset,
                               const int h_start,
                               const int w_start,
                               const uint roll_step,
                               float* unary,
                               float* msg,
                               float* context,
                               float* edge_weights) {
  uint height = param.height, width = param.width;
  uint n_dirs = param.n_dir, n_disp = param.n_disp;
  uint dir = param.dir, dir_inv = param.dir_inv;
  uint batch = param.batch, n_cv = param.n_cv;
  int h_step = param.h_step, w_step = param.w_step;
  bool is_pass_l2r = param.is_pass_l2r;
  float rho = param.rho;
  uint msg_size = batch * n_cv * height * width * n_disp;
  bool enable_seg = (n_disp == 21);

  for (uint i = 0; i <= roll_step; ++i) {
    int current_node_h = h_start + i * h_step;
    int current_node_w = w_start + i * w_step;
    int front_node_h = current_node_h - h_step;
    int front_node_w = current_node_w - w_step;

    if (0 <= current_node_h && current_node_h < (int)height &&
        0 <= front_node_h && front_node_h < (int)height &&
        0 <= current_node_w && current_node_w < (int)width &&
        0 <= front_node_w && front_node_w < (int)width) {
      float min_value = 0;
      uint offset_base = bv_offset + front_node_h * width * n_disp + front_node_w * n_disp;
      uint current_offset_base = bv_offset + current_node_h * width * n_disp + current_node_w * n_disp;
      float norm_min_value = 0;

      for (int current_d = n_disp - 1; current_d >= 0; --current_d) {
        for (int front_d = n_disp - 1; front_d >= 0; --front_d) {
          float context_value = 0;
          if (enable_seg)
            context_value = context[min(current_d, front_d) * n_disp + max(current_d, front_d)];
          else
            context_value = context[std::abs(current_d - front_d)];

          uint offset = offset_base + front_d;
          float msg_sum = 0;

          for (uint k = 0; k < n_dirs; ++k)
            msg_sum += msg[k * msg_size + offset];

          float value = rho * (unary[offset] + msg_sum)
                        - msg[dir_inv * msg_size + offset]
                        + context_value;

          if (front_d == (int)n_disp - 1)
            min_value = value;
          else if (value < min_value)
            min_value = value;
        }

        msg[dir * msg_size + current_offset_base + current_d] = min_value;

        if (current_d == (int)n_disp - 1)
          norm_min_value = min_value;
        else if (min_value < norm_min_value)
          norm_min_value = min_value;
      }

      // Norm
      for (int current_d = n_disp - 1; current_d >= 0; --current_d)
        msg[dir * msg_size + current_offset_base + current_d] -= norm_min_value;
    }
  }
}

void Horizontal(const Param param,
                float* unary,
                float* msg,
                float* context,
                float* edge_weights) {
  uint n_trees = param.n_trees, batch_size = param.batch, n_cv = param.n_cv;
  uint height = param.height, width = param.width, n_disp = param.n_disp;
  int w_step = param.w_step;
  uint roll_step = width;

  for (uint batch = 0; batch < batch_size; ++batch) {
    for (uint volume = 0; volume < n_cv; ++volume) {
      uint bv_offset = (batch * n_cv + volume) * height * width * n_disp;
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (uint tree_id = 0; tree_id < n_trees; ++tree_id) {
        uint h_start = tree_id;
        uint w_start = (w_step > 0) ? 0 : (width - 1);

        DynamicProgramming(param, bv_offset, h_start, w_start, roll_step, unary, msg, context, edge_weights);
      }
    }
  }
}

void DiagonalNarrow(const Param param,
                    float* unary,
                    float* msg,
                    float* context,
                    float* edge_weights) {
  uint n_trees = param.n_trees, batch_size = param.batch, n_cv = param.n_cv;
  uint height = param.height, width = param.width, n_disp = param.n_disp;
  int w_step = param.w_step, h_step = param.h_step;
  uint h_step_abs = std::abs(h_step);
  uint roll_step = (height - 1) / h_step_abs;

  for (uint batch = 0; batch < batch_size; ++batch) {
    for (uint volume = 0; volume < n_cv; ++volume) {
      uint bv_offset = (batch * n_cv + volume) * height * width * n_disp;
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (uint tree_id = 0; tree_id < n_trees; ++tree_id) {
        int tree_id_shift = tree_id - (height - 1) * max(w_step, 0);
        int common1 = tree_id_shift % h_step_abs;
        float common2 = float(tree_id_shift) / float(h_step_abs);
        int h_start = 0, w_start = 0;

        if (w_step > 0) {
          h_start = (h_step_abs - common1) % h_step_abs;
          w_start = ceil(common2);
        } else {
          h_start = common1;
          w_start = floor(common2);
        }

        if (h_step < 0) h_start = height - 1 - h_start;

        DynamicProgramming(param, bv_offset, h_start, w_start, roll_step, unary, msg, context, edge_weights);
      }
    }
  }
}

void DiagonalWide(const Param param,
                  float* unary,
                  float* msg,
                  float* context,
                  float* edge_weights) {
  uint n_trees = param.n_trees, batch_size = param.batch, n_cv = param.n_cv;
  uint height = param.height, width = param.width, n_disp = param.n_disp;
  int w_step = param.w_step, h_step = param.h_step;
  uint h_step_abs = std::abs(h_step);
  uint roll_step = (height - 1) / h_step_abs;

  for (uint batch = 0; batch < batch_size; ++batch) {
    for (uint volume = 0; volume < n_cv; ++volume) {
      uint bv_offset = (batch * n_cv + volume) * height * width * n_disp;
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (uint tree_id = 0; tree_id < n_trees; ++tree_id) {
        int tree_id_shift = tree_id - (height - 1) * max(w_step, 0);
        int h_start = (h_step > 0) ? 0 : (height - 1), w_start = tree_id_shift;

        DynamicProgramming(param, bv_offset, h_start, w_start, roll_step, unary, msg, context, edge_weights);
      }
    }
  }
}

void ForwardCPU(const float rho,
                const int n_iter,
                const at::Tensor unary,
                const at::Tensor context,
                const at::Tensor edge_weights,
                at::Tensor msg,
                at::Tensor cost_final,
                at::Tensor label_all) {
  const uint n_dir = msg.size(0);
  const uint batch = msg.size(1);
  const uint n_cv = msg.size(2);
  const uint height = msg.size(3);
  const uint width = msg.size(4);
  const uint n_disp = msg.size(5);
  float* unary_ptr = unary.data<float>();
  float* context_ptr = context.data<float>();
  float* edge_weight_ptr = nullptr;
  float* msg_ptr = msg.data<float>();  // (n_dir,batch,cv,h,w,n_disp)
  float* cost_final_ptr = cost_final.data<float>();  // (batch,cv,h,w,n_disp)
  uchar* label_all_ptr = nullptr;
  bool is_backward = false, is_training = false;
  bool enable_cal_label = label_all.size(0) == 0 ? false : true;

  if (edge_weights.size(0) != 0)
    edge_weight_ptr = edge_weights.data<float>();

  if (enable_cal_label) label_all_ptr = label_all.data<uchar>();

  std::vector<Param> param_list;
  uint msg_min_size = batch * n_cv * height * width * n_disp;

  for (uint dir = 0; dir < n_dir; ++dir) {
    Param param(n_dir, batch, n_cv, height, width, n_disp, dir, rho, is_backward, is_training);
    UpdateParam(&param);
    param_list.push_back(param);
  }

  for (uint iter = 0; iter < (uint)n_iter; ++iter) {
    for (uint dir = 0; dir < n_dir; ++dir) {
      if (dir < 2)
        Horizontal(param_list[dir], unary_ptr, msg_ptr, context_ptr, edge_weight_ptr);

      if ((2 <= dir) && (dir < 4))
        DiagonalWide(param_list[dir], unary_ptr, msg_ptr, context_ptr, edge_weight_ptr);

      if (4 <= dir) {
        uint h_step_abs = std::abs(param_list[dir].h_step);
        uint w_step_abs = std::abs(param_list[dir].w_step);

        if (h_step_abs > w_step_abs)
          DiagonalNarrow(param_list[dir], unary_ptr, msg_ptr, context_ptr, edge_weight_ptr);
        else
          DiagonalWide(param_list[dir], unary_ptr, msg_ptr, context_ptr, edge_weight_ptr);
      }
    }

    if (enable_cal_label) {
      memcpy(cost_final_ptr, unary_ptr, msg_min_size * sizeof(float));
      CostAggregate(param_list[0], msg_min_size, msg_ptr, cost_final_ptr);
      CalLabel(param_list[0], cost_final_ptr, label_all_ptr + iter * msg_min_size / n_disp);
    }
  }

  memcpy(cost_final_ptr, unary_ptr, msg_min_size * sizeof(float));
  CostAggregate(param_list[0], msg_min_size, msg_ptr, cost_final_ptr);
}

#ifdef USE_HARD
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
             at::Tensor label_all) {
#ifdef USE_CUDA
  CHECK_CONTIGUOUS(unary);
  CHECK_CONTIGUOUS(context);
  CHECK_CONTIGUOUS(edge_weights);
  CHECK_CONTIGUOUS(msg);
  CHECK_CONTIGUOUS(cost_final);
  CHECK_CONTIGUOUS(msg_min_index);
  CHECK_CONTIGUOUS(msg_norm_index);
  CHECK_CONTIGUOUS(unary_update);
  CHECK_CONTIGUOUS(label_all);
#endif

  if ((unary.size(4) > MAX_DISPARITY)) {
    printf("Error!!! Input number of disparity %ld is larger than or not diviable by the preset one %d.\n",
           unary.size(4), MAX_DISPARITY);
    return;
  }

  if (unary.is_cuda()) {
#ifdef USE_CUDA
    ForwardCUDA(rho, n_iter, (bool)enable_min_a_dir, unary, context, edge_weights,
                msg, cost_final, msg_min_index, msg_norm_index, unary_update, label_all);
#endif
  } else
    ForwardCPU(rho, n_iter, unary, context, edge_weights, msg, cost_final, label_all);
}

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
              at::Tensor dunary_update) {
#ifdef USE_CUDA
  CHECK_CONTIGUOUS(context);
  CHECK_CONTIGUOUS(edge_weights);
  CHECK_CONTIGUOUS(dcost_final);
  CHECK_CONTIGUOUS(msg_min_index);
  CHECK_CONTIGUOUS(msg_norm_index);
  CHECK_CONTIGUOUS(dunary);
  CHECK_CONTIGUOUS(dcontext);
  CHECK_CONTIGUOUS(dedge_weights);
  CHECK_CONTIGUOUS(dmsg);
  CHECK_CONTIGUOUS(dunary_update);
#endif

  if (dcost_final.is_cuda()) {
#ifdef USE_CUDA
    BackwardCUDA(rho, context, edge_weights, dcost_final, msg_min_index,
                 msg_norm_index, dunary, dcontext, dedge_weights, dmsg,
                 dunary_update);
#endif
  } else
    printf("Does not support CPU backward.\n");
}
#endif

#ifdef USE_SOFT
// ==================== soft-weighted belief propagation =======================
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
                 at::Tensor label_all) {
#ifdef USE_CUDA
  CHECK_CONTIGUOUS(unary);
  CHECK_CONTIGUOUS(context);
  CHECK_CONTIGUOUS(edge_weights);
  CHECK_CONTIGUOUS(msg);
  CHECK_CONTIGUOUS(msg_edge_label);
  CHECK_CONTIGUOUS(msg_norm_index);
  CHECK_CONTIGUOUS(cost_final);
  CHECK_CONTIGUOUS(unary_update);
  CHECK_CONTIGUOUS(label_all);
#endif

  if ((unary.size(4) > MAX_DISPARITY)) {
    printf("Error!!! Input number of disparity %ld is larger than or not diviable by the preset one %d.\n",
       unary.size(4), MAX_DISPARITY);
    return;
  }

  if (unary.is_cuda()) {
#ifdef USE_CUDA
    ForwardCUDASoft(rho, n_iter, unary, context, edge_weights, msg, msg_edge_label,
                    msg_norm_index, cost_final, unary_update, label_all);
#endif
  } else
    printf("Does not support CPU forward soft.\n");
}

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
                  at::Tensor dunary_update) {
#ifdef USE_CUDA
  CHECK_CONTIGUOUS(dcost_final);
  CHECK_CONTIGUOUS(context);
  CHECK_CONTIGUOUS(edge_weights);
  CHECK_CONTIGUOUS(msg_edge_label);
  CHECK_CONTIGUOUS(msg_norm_index);
  CHECK_CONTIGUOUS(dunary);
  CHECK_CONTIGUOUS(dcontext);
  CHECK_CONTIGUOUS(dedge_weights);
  CHECK_CONTIGUOUS(dmsg);
  CHECK_CONTIGUOUS(dunary_update);
#endif

  if (dcost_final.is_cuda()) {
#ifdef USE_CUDA
    BackwardCUDASoft(rho, dcost_final, context, edge_weights, msg_edge_label,
                     msg_norm_index, dunary, dcontext, dedge_weights, dmsg,
                     dunary_update);
#endif
  } else
    printf("Does not support CPU backward soft.\n");
}

void SoftWeightedMessage(at::Tensor message,
                         at::Tensor message_soft_sum_norm,
                         at::Tensor message_soft_sum_min_ind) {
#ifdef USE_CUDA
  CHECK_CONTIGUOUS(message);
  CHECK_CONTIGUOUS(message_soft_sum_norm);
  CHECK_CONTIGUOUS(message_soft_sum_min_ind);
#endif

  if (message.is_cuda()) {
#ifdef USE_CUDA
    SoftWeightedMessageCUDA(message, message_soft_sum_norm, message_soft_sum_min_ind);
#endif
  } else
    printf("Does not support CPU soft weighted message.\n");
}

void SoftWeightedMessageBack(const at::Tensor dmessage_soft_sum_norm,
                             const at::Tensor message,
                             const at::Tensor message_soft_sum_min_ind,
                             at::Tensor dmessage) {
#ifdef USE_CUDA
  CHECK_CONTIGUOUS(dmessage_soft_sum_norm);
  CHECK_CONTIGUOUS(message);
  CHECK_CONTIGUOUS(message_soft_sum_min_ind);
  CHECK_CONTIGUOUS(dmessage);
#endif

  if (message.is_cuda()) {
#ifdef USE_CUDA
    SoftWeightedMessageBackCUDA(dmessage_soft_sum_norm, message, message_soft_sum_min_ind, dmessage);
#endif
  } else
    printf("Does not support CPU soft weighted message backward.\n");
}
#endif

#ifdef __cplusplus
  }
#endif

#if defined(USE_HARD) && !defined(USE_SOFT)
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &Forward, "TRWP forward (CUDA)");
    m.def("backward", &Backward, "TRWP backward (CUDA)");
  }
#elif !defined(USE_HARD) && defined(USE_SOFT)
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_soft", &ForwardSoft, "TRWP soft forward (CUDA)");
    m.def("backward_soft", &BackwardSoft, "TRWP soft backward (CUDA)");
    m.def("soft_weighted_message", &SoftWeightedMessage, "SoftWeightedMessage (CUDA)");
    m.def("soft_weighted_message_back", &SoftWeightedMessageBack, "SoftWeightedMessage backward (CUDA)");
  }
#elif defined(USE_HARD) && defined(USE_SOFT)
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &Forward, "TRWP forward (CUDA)");
    m.def("backward", &Backward, "TRWP backward (CUDA)");
    m.def("forward_soft", &ForwardSoft, "TRWP soft forward (CUDA)");
    m.def("backward_soft", &BackwardSoft, "TRWP soft backward (CUDA)");
    m.def("soft_weighted_message", &SoftWeightedMessage, "SoftWeightedMessage (CUDA)");
    m.def("soft_weighted_message_back", &SoftWeightedMessageBack, "SoftWeightedMessage backward (CUDA)");
  }
#endif

