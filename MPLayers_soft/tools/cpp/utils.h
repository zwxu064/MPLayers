#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cctype>
#include <sys/time.h>

#ifdef USE_EIGEN
  #include <Eigen/Eigen>
#endif

using namespace std;

#define MAX_VALUE 9999
#define NUM_THREADS 16

typedef unsigned char uchar;
typedef unsigned int uint;

#ifdef USE_FLOAT
  #define CostType float
#else
  #define CostType double
#endif

#ifdef USE_EIGEN
  using namespace Eigen;
  typedef Matrix<uchar, Dynamic, Dynamic> MatrixXuc;
  typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
  typedef Matrix<int, Dynamic, Dynamic, RowMajor> RowMatrixXi;
  typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMatrixXf;
  typedef Matrix<uchar, Dynamic, Dynamic, RowMajor> RowMatrixXuc;
  typedef Matrix<uchar, 1, Dynamic> VectorXuc;

  #ifdef USE_FLOAT
    #define RowMatrixXCost RowMatrixXf
    #define VectorXCost VectorXf
  #else
    #define RowMatrixXCost RowMatrixXd
    #define VectorXCost VectorXd
  #endif
#endif

double MyTime(const timeval &time_start);

int ReadLine(FILE *infile,
             char *buf,
             int &lineno);

template <typename T>
bool ReadFromFile(const char *fname,
                  T *unary);

template <typename T>
T** allocate2D(int row,
               int col,
               T value);

template <typename T>
void release2D(T **data,
               int row);

template <typename T>
void CopyVector(T *to,
                T *from,
                int size);

template <typename T>
void CopyVector2D(T **to,
                  T **from,
                  int offsetVar,
                  int rowOffset,
                  int row,
                  int col);

template <typename T>
T SubtractMin(T *data,
              int size,
              uchar *minIndex);

template <typename T>
void print2D(T **data,
             int row,
             int col,
             string info);

template <typename T>
T FindMin(int size,
          T *data,
          uchar *minIndex);

template <typename T>
void FindMin2D(int dir,
               int rows,
               int cols,
               T **data,
               T *result,
               uchar *minIndex);

FILE *CreateLogFile(string filePath);

#endif // _UTILS_HPP_
