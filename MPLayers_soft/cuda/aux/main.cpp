#include <stdio.h>
#include <iostream>
#include <fstream>
#include "utils.h"
#include "aux.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  if (argc < 9) {
    printf("Error [prog] [nlabels] [imLPath] [imRPath] "
           "[unaryPath] [pairwisePath] [rgbPath] [gradThresh] "
           "[gradPenalty].\n");
    return -1;
  }

  int nLabels = atoi(argv[1]);
  char *imLPath = argv[2];
  char *imRPath = argv[3];
  char *unaryPath = argv[4];
  char *pairwisePath = argv[5];
  char *rgbPath = argv[6];
  int gradThresh = atoi(argv[7]);
  int gradPenalty = atoi(argv[8]);
  // int gradThresh = 8, gradPenalty = 2;
  int birchfield = 1, squaredDiffs= 0, truncDiffs = 255;

  // Calculate unary
  printf("Creating unary file.\n");
  Mat im1 = imread(imLPath), im2 = imread(imRPath);
  if (im1.rows != im2.rows || im1.cols != im2.cols) {
    printf("image size do not match, im1: h:%d, w:%d; im2: h:%d, w:%d.\n",
           im1.rows, im1.cols, im2.rows, im2.cols);
    return -1;
  }
  int width = im1.cols, height = im1.rows, nChannels = im1.channels();
  int nNodes = height * width;

  CostType *dsi = new CostType[nNodes * nLabels];
  computeDSI(im1, im2, dsi, nLabels, birchfield, squaredDiffs,
                       truncDiffs);

  fstream file;
  file.open(unaryPath, ios_base::out);
  for (int h = 0; h < height; ++h)
    for (int w = 0; w < width; ++w)
      for (int l = 0; l < nLabels; ++l) {
        int id = h * width + w;
        file << h << "," << w << "," << l << "," << dsi[id * nLabels + l]
                << "\n";
      }
  file.close();
  delete [] dsi;

  // Calculate pairwise
  printf("Creating pairwise file.\n");
  CostType *hc = new CostType[nNodes];
  CostType *vc = new CostType[nNodes];
  computeCues(im1, hc, vc, gradThresh, gradPenalty);

  file.open(pairwisePath, ios_base::out);
  for (int h = 0; h < height; ++h)
    for (int w = 0; w < width; ++w) {
      int id = h * width + w;
      int hValue = -1, vValue = -1;
      if (w < width - 1) hValue = hc[id];
      if (h < height - 1) vValue = vc[id];
      file << h << "," << w << "," << hValue << "," << vValue << "\n";
    }
  file.close();
  delete [] hc;
  delete [] vc;

  // RGB
  printf("Creating RGB file.\n");
  file.open(rgbPath, ios_base::out);
  for (int h = 0; h < height; ++h)
    for (int w = 0; w < width; ++w) {
      uchar *pix = im1.ptr(h, w);
      for (int c = 0; c < nChannels; ++c) {
        CostType value = static_cast<CostType>(pix[c]);
        file << h << "," << w << "," << c << "," << value << "\n";
      }
    }
  file.close();
}
