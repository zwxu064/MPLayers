#ifndef AUX_H
#define AUX_H

#include <stdio.h>
#include <iostream>
#include "utils.h"

#ifdef USE_OPENCV
  #include <opencv2/core/core.hpp>
  #include <opencv2/highgui/highgui.hpp>
  #include <opencv2/imgcodecs.hpp>
  using namespace cv;
#else
  #include "imageLib.h"
#endif

using namespace std;

template <typename T>
void computeDSI(T im1,
                T im2,
                CostType *dsi,
                int nLabels,
                int birchfield,
                int squaredDiffs,
                int truncDiffs);

template <typename T>
void computeCues(T im1,
                 CostType *hCue,
                 CostType *vCue,
                 int gradThresh,
                 int gradPenalty);

void computeSmoothnessContext(int nLabels,
                              string method,
                              CostType smoothnessMax,
                              CostType **smoothnessContext);

#endif
