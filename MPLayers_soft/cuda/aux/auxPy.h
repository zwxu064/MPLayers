#ifndef AUXPY_H
#define AUXPY_H

#include <stdio.h>
#include <iostream>

#if TORCH_VERSION_MAJOR == 0
  #include <torch/torch.h>  // pytorch 0.4.1, cuda 8
#else
 #include <torch/extension.h>  // pytorch 1.1.0, cuda 10
#endif

#include <ATen/ATen.h>
#include "aux.h"
#include "utils.h"

void ComputerAllTerms(const char* leftImgPath,
                      const char* rightImgPath,
                      const char* pMethod,
                      const int gradThresh,
                      const int gradPenalty,
                      const int truncatedValue,
                      const int height,
                      const int width,
                      const int nChannels,
                      const int nLabels,
                      at::Tensor dataCost,
                      at::Tensor RGB,
                      at::Tensor hCue,
                      at::Tensor vCue,
                      at::Tensor smoothnessContext);
#endif
