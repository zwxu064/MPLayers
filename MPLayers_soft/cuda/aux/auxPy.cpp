#include <pybind11/pybind11.h>
#include "auxPy.h"

namespace py = pybind11;
using namespace std;

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
                      at::Tensor dataCostT,
                      at::Tensor RGBT,
                      at::Tensor hCueT,
                      at::Tensor vCueT,
                      at::Tensor smoothnessContextT) {
  printf("Running compute all terms ...\n");

  int birchfield = 1, squaredDiffs= 0, truncDiffs = 255;
  string method(pMethod);
  CostType smoothnessMax = (CostType)truncatedValue;

#ifdef USE_OPENCV
  Mat im1 = imread(leftImgPath, IMREAD_UNCHANGED), im2 = imread(rightImgPath, IMREAD_UNCHANGED);
  if (height != im2.rows || width != im2.cols)
    printf("size does not match, exter: h:%d, w:%d, inter: h:%d, w:%d.\n",
           height, width, im2.rows, im2.cols);
#else
  int verbose = 1;
  CByteImage im1, im2;    // input images (gray or color)
  ReadImageVerb(im1, leftImgPath, verbose);
  ReadImageVerb(im2, rightImgPath, verbose);

  CShape sh = im1.Shape();
  CShape sh2 = im2.Shape();

  if (sh != sh2)
    throw CError("image shapes don't match");

  if (width != sh.width || height == sh.height)
    printf("size does not match, exter: h:%d, w:%d, inter: h:%d, w:%d.\n",
           height, width, sh.height, sh.width);
#endif

  int nNodes = height * width;

#ifdef USE_OPENCV
  if (nChannels != min(3, im2.channels()))
    printf("channels do not match, exter: %d, inter: %d.\n",
           nChannels, im2.channels());
#else
  if (nChannels != min(3, sh.nBands))
    printf("channels do not match, exter: %d, inter: %d.\n",
           nChannels, sh.nBands);
#endif

  // Unary cost
  CostType *dsi = new CostType[nNodes * nLabels];
  computeDSI(im1, im2, dsi, nLabels, birchfield, squaredDiffs, truncDiffs);
  CopyVector<CostType>(dataCostT.data<CostType>(), dsi, nNodes * nLabels);
  delete [] dsi;

  // RGB
  CostType *RGB = new CostType[nNodes * nChannels];
  for (int h = 0; h < height; ++h)
    for (int w = 0; w < width; ++w) {
      int nodeId = h * width + w;
#ifdef USE_OPENCV
      uchar *pix = im1.ptr(h, w);
#else
      uchar *pix = &im1.Pixel(w, h, 0);
#endif
      for (int j = 0; j < nChannels; ++j)
        RGB[nodeId * nChannels + j] = static_cast<CostType>(pix[j]);
    }
  CopyVector<CostType>(RGBT.data<CostType>(), RGB, nNodes * nChannels);
  delete [] RGB;

  // Cues
  CostType *hc = new CostType[nNodes];
  CostType *vc = new CostType[nNodes];
  CostType *hcVector = new CostType[height * (width - 1)];
  CostType *vcVector = new CostType[(height - 1) * width];
  computeCues(im1, hc, vc, gradThresh, gradPenalty);
  for (int h = 0; h < height; ++h)
    for (int w = 0; w < width; ++w) {
      int hCueId = h * (width - 1) + w;
      int vCueId = w * (height - 1) + h;
      int id = h * width + w;
      if (w < width - 1) hcVector[hCueId] = hc[id];
      if (h < height - 1) vcVector[vCueId] = vc[id];
  }
  CopyVector(hCueT.data<CostType>(), hcVector, height * (width - 1));
  CopyVector(vCueT.data<CostType>(), vcVector, (height - 1) * width);
  delete [] hcVector;
  delete [] vcVector;
  delete [] hc;
  delete [] vc;

  // Smoothness context
  CostType **smoothnessContext = allocate2D<CostType>(nLabels, nLabels, 0);
  computeSmoothnessContext(nLabels, method, smoothnessMax, smoothnessContext);
  CopyVector<CostType>(smoothnessContextT.data<CostType>(), smoothnessContext[0],
                       nLabels * nLabels);
  release2D<CostType>(smoothnessContext, nLabels);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_all_terms", &ComputerAllTerms, "message passing terms");
}
