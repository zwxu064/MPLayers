#include "common.h"

#ifdef __cplusplus
  extern "C" {
#endif

uint GetNumThreadATree(const uint in_size,
                       const uint space) {
  return uint((in_size + space - 1) / space) * space;
}

uint GetNumBlock(const uint n_threads,
                 const uint n_thread_a_block) {
  return uint(n_threads + n_thread_a_block - 1) / n_thread_a_block;
}

void UpdateParam(Param* param) {
  uint dir = param->dir, dir_inv = 0, n_trees = 0;
  int h_step = 0, w_step = 0;
  bool is_pass_l2r = (dir % 2 == 0);
  uint height = param->height, width = param->width;

  if (is_pass_l2r)
    dir_inv = dir + 1;
  else
    dir_inv = dir - 1;

  if (dir == 0) {
    h_step = 0;
    w_step = 1;
  } else if (dir == 1) {
    h_step = 0;
    w_step = -1;
  } else if (dir == 2) {
    h_step = 1;
    w_step = 0;
  } else if (dir == 3) {
    h_step = -1;
    w_step = 0;
  } else if (dir == 4) {
    h_step = 1;
    w_step = 1;
  } else if (dir == 5) {
    h_step = -1;
    w_step = -1;
  } else if (dir == 6) {
    h_step = 1;
    w_step = -1;
  } else if (dir == 7) {
    h_step = -1;
    w_step = 1;
  } else if (dir == 8) {
    h_step = 1;
    w_step = 2;
  } else if (dir == 9) {
    h_step = -1;
    w_step = -2;
  } else if (dir == 10) {
    h_step = 2;
    w_step = 1;
  } else if (dir == 11) {
    h_step = -2;
    w_step = -1;
  } else if (dir == 12) {
    h_step = 2;
    w_step = -1;
  } else if (dir == 13) {
    h_step = -2;
    w_step = 1;
  } else if (dir == 14) {
    h_step = 1;
    w_step = -2;
  } else if (dir == 15) {
    h_step = -1;
    w_step = 2;
  }

  if (dir <= 1) n_trees = height;
  else if (dir <= 3) n_trees = width;
  else if (dir <= 7) n_trees = height + width - 1;
  else {
    uint h_step_abs = std::abs(h_step), w_step_abs = std::abs(w_step);
    if (h_step_abs > w_step_abs)
      n_trees = height + (width - 1) * h_step_abs;
    else
      n_trees = width + (height - 1) * w_step_abs;
  }

  if (param->enable_backward) {
    h_step = -h_step;
    w_step = -w_step;
  }

  int h_step_abs = abs(h_step), w_step_abs = abs(w_step);

  param->h_step = h_step;
  param->w_step = w_step;
  param->dir_inv = dir_inv;
  param->is_pass_l2r = is_pass_l2r;
  param->n_trees = n_trees;
  param->n_thread_a_tree = GetNumThreadATree(param->n_disp, WARP_SIZE);
  param->dir_weight = sqrt(h_step_abs * h_step_abs + w_step_abs * w_step_abs);
}

void CostAggregate(const Param param,
                   const uint msg_min_size,
                   float* msg,
                   float* cost_final) {
  uint batch_size = param.batch, n_cv = param.n_cv;
  uint height = param.height, width = param.width;
  uint n_disp = param.n_disp, n_dir = param.n_dir;
  uint msg_size = batch_size * n_cv * height * width * n_disp;

  for (uint batch = 0; batch < batch_size; ++batch) {
    for (uint volume = 0; volume < n_cv; ++volume) {
      uint bv_offset = (batch * n_cv + volume) * height * width * n_disp;

#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (uint i = 0; i < height * width; ++i) {
        uint h = i / width, w = i % width;

        for (uint d = 0; d < n_disp; ++d) {
          uint offset = bv_offset + h * width * n_disp + w * n_disp + d;
          float msg_sum = 0;

          for (uint k = 0; k < n_dir; ++k) {
            msg_sum += msg[k * msg_size + offset] / param.dir_weight;
          }

          cost_final[offset] += msg_sum;
        }
      }
    }
  }
}

void CalLabel(const Param param,
              const float* final_cost,
              uchar* label_all) {
  uint n_disp = param.n_disp;
  uint batch = param.batch, n_cv = param.n_cv;
  uint height = param.height, width = param.width;

  for (uint bv = 0; bv < batch * n_cv; ++bv) {
    uint offset_labeling = bv * height * width;
    uint offset_cost = bv * height * width * n_disp;

#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for (uint node = 0; node < height * width; ++node) {
      uchar label = 0;
      float min_value = final_cost[offset_cost + node * n_disp];
      for (uchar l = 0; l < (uchar)n_disp; ++l) {
        float value = final_cost[offset_cost + node * n_disp + l];
        if (value < min_value) {
          min_value = value;
          label = l;
        }
      }
      label_all[offset_labeling + node] = label;
    }
  }

//double unary_energy = 0, pairwise_energy = 0;
//  for (uint bv = 0; bv < batch * n_cv; ++bv) {
//    uint offset_labeling = bv * height * width;
//    uint offset_cost = bv * height * width * n_disp;
//    unary_energy = pairwise_energy = 0;
//
//    for (uint node = 0; node < height * width; ++node) {
//      uint h = node / width, w = node % width;
//      unary_energy += unary[offset_cost + node * n_disp + labeling[offset_labeling + node]];
//
//      // Vertical
//      if (h < height - 1) {
//        uint node_down = node + width;
//        pairwise_energy += context[labeling[offset_labeling + node] * n_disp
//                           + labeling[offset_labeling + node_down]];
//      }
//
//      // Horizontal
//      if (w < width - 1) {
//        uint node_right = node + 1;
//        pairwise_energy += context[labeling[offset_labeling + node] * n_disp
//                           + labeling[offset_labeling + node_right]];
//      }
//    }
//
//    energy_all[bv] = (float)(unary_energy + pairwise_energy);
//  }
}

#ifdef __cplusplus
  }
#endif
