#ifndef __COMMON_H__
#define __COMMON_H__

#include <math.h>
#include <stdio.h>
#include <iostream>
#include "utils.h"

#ifdef __cplusplus
  extern "C" {
#endif

// Define MAX_DISPARITY in setup_*.py for stereo and segmentation
// #define MAX_DISPARITY 32  // set 32 for segmentation (21 classes) or 64 for stereo for (48 disparisities)
#define MAX_THREADS_PER_BLOCK 1024  // 256
#define MAX_BLOCKS_PER_GRID 65535
#define WARP_SIZE 32
#define MAX_WARP_GROUP ((MAX_DISPARITY + WARP_SIZE - 1) / WARP_SIZE)
#define MAX_NUM_WARPS (MAX_THREADS_PER_BLOCK / WARP_SIZE)
#define FULL_MASK 0xffffffff
#define USE_MSGNORM_NAIVE
// #define CUDA_ERROR_CHECK
#define MAX_PARALLEL_DISPS ((MAX_THREADS_PER_BLOCK) / WARP_SIZE)  // for shared mem
#define MAX_SHARED_MEM_PER_BLOCK MAX_THREADS_PER_BLOCK

struct Param {
  Param(uint n_dir, uint batch, uint n_cv, uint height, uint width, uint n_disp,
  uint dir, float rho, bool enable_backward, bool is_training, bool enable_sgm=false)
  : n_dir(n_dir), batch(batch), n_cv(n_cv), height(height), width(width),
  n_disp(n_disp), dir(dir), rho(rho), enable_backward(enable_backward),
  is_training(is_training), enable_sgm(enable_sgm) {};

  void SetParam(bool enable_backward_new) {
    if (enable_backward_new != enable_backward) {
      enable_backward = enable_backward_new;
      h_step = -h_step;
      w_step = -w_step;
    }
  }

  uint n_dir = 0;
  uint batch = 0;
  uint n_cv = 0;
  uint height = 0;
  uint width = 0;
  uint n_disp = 0;
  int h_step = 0;
  int w_step = 0;
  uint dir = 0;
  uint dir_inv = 0;
  uint n_trees = 0;
  uint n_thread_a_tree = 0;
  float rho = 1;
  float dir_weight = 1;
  bool enable_backward = false;
  bool is_training = false;
  bool is_pass_l2r = true;
  bool enable_min_a_dir = false;
  bool enable_sgm = false;
};

void CostAggregate(const Param param,
                   const uint msg_min_size,
                   float* msg,
                   float* cost_final);

void CalLabel(const Param param,
              const float* final_cost,
              uchar* label_all);

uint GetNumThreadATree(const uint in_size,
                       const uint space);

uint GetNumBlock(const uint n_threads,
                 const uint n_thread_a_block);

void UpdateParam(Param* param);

#ifdef __cplusplus
  }
#endif

#endif  // __COMMON_H__
