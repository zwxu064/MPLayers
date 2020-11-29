#include "aux.h"

using namespace std;

template <typename T>
void computeDSI(T im1,     // source (reference) image
                T im2,     // destination (match) image
                CostType *dsi,      // computed cost volume
                int nLabels,        // number of disparities
                int birchfield,     // use Birchfield/Tomasi costs
                int squaredDiffs,   // use squared differences
                int truncDiffs) {   // truncated differences
#if USE_OPENCV
  int width = im1.cols, height = im1.rows, nB = im1.channels();
#else
  CShape sh = im1.Shape();
  int width = sh.width, height = sh.height, nB = sh.nBands;
#endif
  // dsi = new CostType[width * height * nLabels];

  int nColors = min(3, nB);

  // worst value for sumdiff below
  int worst_match = nColors * (squaredDiffs ? 255 * 255 : 255);
  // truncation threshold - NOTE: if squared, don't multiply by nColors
  // (Eucl. dist.)
  int maxsumdiff = squaredDiffs ? truncDiffs * truncDiffs :
                   nColors * abs(truncDiffs);
  // value for out-of-bounds matches
  int badcost = min(worst_match, maxsumdiff);

  int dsiIndex = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
#if USE_OPENCV
      uchar *pix1 = im1.ptr(y, x);
#else
      uchar *pix1 = &im1.Pixel(x, y, 0);
#endif
      for (int d = 0; d < nLabels; d++) {
        int x2 = x-d;
        int dsiValue;

        if (x2 >= 0 && d < nLabels) { // in bounds
#if USE_OPENCV
          uchar *pix2 = im2.ptr(y, x2);
#else
          uchar *pix2 = &im2.Pixel(x2, y, 0);
#endif
          int sumdiff = 0;
          for (int b = 0; b < nColors; b++) {
            int diff = 0;
            if (birchfield) {
              // Birchfield/Tomasi cost
              int im1c = pix1[b];
              int im1l = x == 0?   im1c : (im1c + pix1[b - nB]) / 2;
              int im1r = x == width-1? im1c : (im1c + pix1[b + nB]) / 2;
              int im2c = pix2[b];
              int im2l = x2 == 0?   im2c : (im2c + pix2[b - nB]) / 2;
              int im2r = x2 == width-1? im2c : (im2c + pix2[b + nB]) / 2;
              int min1 = min(im1c, min(im1l, im1r));
              int max1 = max(im1c, max(im1l, im1r));
              int min2 = min(im2c, min(im2l, im2r));
              int max2 = max(im2c, max(im2l, im2r));
              int di1 = max(0, max(im1c - max2, min2 - im1c));
              int di2 = max(0, max(im2c - max1, min1 - im2c));
              diff = min(di1, di2);
            } else {
              // simple absolute difference
              int di = pix1[b] - pix2[b];
              diff = abs(di);
            }
            // square diffs if requested (Birchfield too...)
            sumdiff += (squaredDiffs ? diff * diff : diff);
          }

          // truncate diffs
          dsiValue = min(sumdiff, maxsumdiff);
              } else { // out of bounds: use maximum truncated cost
          dsiValue = badcost;
        }
        //int x0=-140, y0=-150;
        //if (x==x0 && y==y0)
        //  printf("dsi(%d,%d,%2d)=%3d\n", x, y, d, dsiValue);

        // The cost of pixel p and label l is stored at dsi[p*nLabels+l]
        dsi[dsiIndex++] = static_cast<CostType>(dsiValue);
      }
    }
  }
}

template <typename T>
void computeCues(T im1,
                 CostType *hCue,
                 CostType *vCue,
                 int gradThresh,
                 int gradPenalty) {
#ifdef USE_OPENCV
  int width = im1.cols, height = im1.rows, nB = im1.channels();
#else
  CShape sh = im1.Shape();
  int width = sh.width, height = sh.height, nB = sh.nBands;
#endif
  // hCue = new CostType[width * height];
  // vCue = new CostType[width * height];

  int nColors = min(3, nB);

  // below we compute sum of squared colordiffs, so need to adjust threshold
  // accordingly (results in RMS)
  gradThresh *= nColors * gradThresh;

#ifdef USE_OPENCV
  Mat hc(height, width, CV_8UC1), vc(height, width, CV_8UC1);
#else
  sh.nBands=1;
  CByteImage hc(sh), vc(sh);
#endif
  int n = 0;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
#ifdef USE_OPENCV
      uchar *pix   = im1.ptr(y, x);
      uchar *pix1x = im1.ptr(y, x + (x < width-1));
      uchar *pix1y = im1.ptr(y + (y < height-1), x);
#else
      uchar *pix   = &im1.Pixel(x, y, 0);
      uchar *pix1x = &im1.Pixel(x + (x < width-1), y, 0);
      uchar *pix1y = &im1.Pixel(x, y + (y < height-1), 0);
#endif
      int sx = 0, sy = 0;

      for (int b = 0; b < nColors; b++) {
        int dx = pix[b] - pix1x[b];
        int dy = pix[b] - pix1y[b];
        sx += dx * dx;
        sy += dy * dy;
      }

      hCue[n] = (sx < gradThresh ? gradPenalty : 1);
      vCue[n] = (sy < gradThresh ? gradPenalty : 1);

#ifdef USE_OPENCV
      hc.ptr(y, x)[0] = 100 * hCue[n];
      vc.ptr(y, x)[0] = 100 * vCue[n];
#else
      hc.Pixel(x, y, 0) = 100*hCue[n];
      vc.Pixel(x, y, 0) = 100*vCue[n];
#endif
      n++;
    }
  }

// #ifdef USE_OPENCV
//   imwrite("hcue.png", hc);
//   imwrite("vcue.png", vc);
// #else
//   WriteImageVerb(hc, "hcue.png", true);
//   WriteImageVerb(vc, "vcue.png", true);
// #endif
}

void computeSmoothnessContext(int nLabels,
                              string method,
                              CostType smoothnessMax,
                              CostType **smoothnessContext) {
  if (method == "L") {  // L1
    for (int xj = 0; xj < nLabels; ++xj)
      for (int xi = 0; xi < nLabels; ++xi)
        smoothnessContext[xj][xi] = static_cast<CostType>(abs(xj - xi));
  } else if (method == "Q") {  // L2
    smoothnessMax *= smoothnessMax;
    for (int xj = 0; xj < nLabels; ++xj)
      for (int xi = 0; xi < nLabels; ++xi) {
        CostType value = static_cast<CostType>(xj - xi);
        smoothnessContext[xj][xi] = value * value;
      }
  } else if (method == "TL") {
    for (int xj = 0; xj < nLabels; ++xj)
      for (int xi = 0; xi < nLabels; ++xi) {
        CostType value = static_cast<CostType>(abs(xj - xi));
        smoothnessContext[xj][xi] = min(value, smoothnessMax);
      }
  } else if (method == "TQ") {
    smoothnessMax *= smoothnessMax;
    for (int xj = 0; xj < nLabels; ++xj)
      for (int xi = 0; xi < nLabels; ++xi) {
        CostType value = static_cast<CostType>(xj - xi);
        smoothnessContext[xj][xi] = min(value * value, smoothnessMax);
      }
  } else if (method == "Cauchy") {
    for (int xj = 0; xj < nLabels; ++xj)
      for (int xi = 0; xi < nLabels; ++xi) {
        CostType value = static_cast<CostType>(xj - xi);
        CostType valueTmp = value / static_cast<double>(smoothnessMax);
        smoothnessContext[xj][xi] = 0.5 * smoothnessMax * smoothnessMax *
                                    log(1 + valueTmp * valueTmp);
      }
  } else if (method == "Huber") {
    for (int xj = 0; xj < nLabels; ++xj)
      for (int xi = 0; xi < nLabels; ++xi) {
        CostType value = static_cast<CostType>(xj - xi);
        if (abs(value) <= smoothnessMax)
          smoothnessContext[xj][xi] = value * value;
        else
          smoothnessContext[xj][xi] = 2 * smoothnessMax * abs(value) -
                                      smoothnessMax * smoothnessMax;
      }
  } else {
    printf("%s to be continued.\n", method.c_str());
  }
}

#ifdef USE_OPENCV
template void computeDSI<Mat>(Mat im1, Mat im2, CostType *dsi, int nLabels,
                              int birchfield, int squaredDiffs, int truncated);
template void computeCues<Mat>(Mat im1, CostType *hCue, CostType *vCue,
                              int gradThresh, int gradPenalty);
#else
template void computeDSI<CByteImage>(CByteImage im1, CByteImage im2,
                                     CostType *dsi, int nLabels, int birchfield,
                                     int squaredDiffs, int truncated);
template void computeCues<CByteImage>(CByteImage im1, CostType *hCue,
                                      CostType *vCue, int gradThresh,
                                      int gradPenalty);
#endif
