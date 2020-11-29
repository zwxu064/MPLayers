#include "utils.h"

double MyTime(const timeval &time_start) {
    timeval current;
    gettimeofday(&current, nullptr);
    return (current.tv_sec - time_start.tv_sec) * 1e6
            + (current.tv_usec - time_start.tv_usec);
}

int ReadLine(FILE *infile,
             char *buf,
             int &lineno) {
    // Reads a line into a file.
    // It skips comment lines, starting with '#'
    // and also blank lines.  It strips leading blanks.

    // Increment the line number
    lineno++;

    int len = 0;

    while (1) {
        // Read a character
        int n = getc(infile);

        // If we are at EOF, then stop
        if (n == EOF) {
            // This is the normal end of the file
            if (len == 0) return EOF;

            // Here, the file ends before end-of-line
            buf[len] = (char)0;
            return len;
        }

        // If, we find end-of-line then return
        if (n == '\n') {
            if (len == 0) {
                lineno++;
                continue;  // Skips blank lines
            }

            // If it is a comment line then continue
            if (buf[0] == '#') {
                len = 0;
                lineno++;
                continue;
            }

            // Otherwise, terminate and return
            buf[len] = (char)0;
            return len;
        }

        // It is now a genuine character, but skip leading spaces
        if (len == 0 && isspace(n)) continue;

        // We have an ordinary character, then transfer it
        buf[len++] = n;
    }
}

template <typename T>
bool ReadFromFile(const char *fname,
                  T *unary) {
    // First open the file
    FILE *infile = fopen(fname, "r");
    if (!infile) {
        printf("Error, cannot open file \"%s\" for input", fname);
        return false;
    }

    // Set line counter to zero
    int lineno = 0;

    // Now, read it line by line
    char linebuf[1024] = {};  // Excessive, initilization by {}

    // Read the number of pixels and nodes
    int nclasses = 0, height = 0, width = 0;
    int len = ReadLine(infile, linebuf, lineno);

    if (len == EOF) {
        printf("Error, premature end of file \"%s\", line %d", fname, lineno);
        return false;
    }

    // Extract npixels and nlabels from this line
    int n      = sscanf(linebuf, "%d %d %d", &height, &width, &nclasses);
    int nnodes = height * width;

    if (n != 3) {
        printf("Error, reading nnodes in file \"%s\", line %d", fname, lineno);
        return false;
    }

    // Read all the unary energies
    for (int i = 0; i < nnodes * nclasses; i++) {
        // Read a line
        memset(linebuf, 0, sizeof(char) * sizeof(linebuf));
        int len = ReadLine(infile, linebuf, lineno);

        // If at EOF, then this file has terminated prematurely
        if (len == EOF) {
            printf("Error, unexpected EOF reading unary energies from file \"%s\"", fname);
            return false;
        }

        // Otherwise, we have a good line
        // Read the format
        int pixel;
        double label, denergy;
        int n = sscanf(linebuf, "%u %lf %lf", &pixel, &label, &denergy);
        if (n != 3) {
            printf("Error, too few values in line %d of file \"%s\"", lineno, fname);
            return false;
        }

        if (pixel < 0 || pixel >= height * width || label < 0 || label >= nclasses) {
            printf("Error, invalid pixel index: %u or label index: %lf", pixel, label);
            return false;
        }

        // Now, make a record and store it
        unary[i] = static_cast<T>(denergy);
    }

    fclose(infile);
    // At present if it fails, it just terminates - so if it gets here, all good.

    return true;
}

template <typename T>
T** allocate2D(int row, int col, T value) {
  // T **data = new T*[row];
  // for (int i = 0; i < row; ++i) {
  //   data[i] = new T[col];
  //   if (value == 0)
  //     memset(data[i], value, sizeof(T) * col);
  //   else {
  //     for (int j = 0; j < col; ++j)
  //       data[i][j] = value;
  //   }
  // }

  // Continuous
  T **data = new T*[row];
  data[0] = new T[row * col];
  for (int i = 1; i < row; ++i) data[i] = data[i - 1] + col;

  if (value == 0)
    memset(data[0], 0, sizeof(T) * row * col);
  else
    for (int i = 0; i < row * col; ++i)
      *(data[0] + i) = value;

  return data;
}

template <typename T>
void release2D(T **data, int row) {
  if (data == nullptr) return;

  // for (int i = 0; i < row; ++i) {
  //   delete [] data[i];
  //   data[i] = nullptr;
  // }

  // Continuous
  delete [] data[0];
  data[0] = nullptr;

  delete [] data;
  data = nullptr;
}

template <typename T>
void CopyVector(T *to,
                T *from,
                int size) {
  memcpy(to, from, sizeof(T) * size);
}

template <typename T>
void CopyVector2D(T **to,
                  T **from,
                  int offsetVar,
                  int rowOffset,
                  int row,
                  int col) {
  for (int j = 0; j < row; ++j)
    if (offsetVar == 0) memcpy(to[j + rowOffset], from[j], sizeof(T) * col);
    else memcpy(to[j], from[j + rowOffset], sizeof(T) * col);
}

template <typename T>
T SubtractMin(T *data,
              int size,
              uchar *minIndex) {
  T minValue = data[0];
  if (minIndex != nullptr)  *minIndex = 0;

  for (int i = 1; i < size; ++i) {
    T value = data[i];
    if (value < minValue) {
      minValue = value;
      if (minIndex != nullptr) *minIndex = i;
    }
  }

  for (int i = 0; i < size; ++i)
    data[i] -= minValue;

  return minValue;
}

template <typename T>
void print2D(T **data, int row, int col, string info) {
  printf("%s\n", info.c_str());
  for (int h = 0; h < row; ++h) {
    for (int w = 0; w < col; ++w)
      printf("  %.4f ", static_cast<float>(data[h][w]));
    printf("\n");
  }
}

template <typename T>
T FindMin(int size,
          T *data,
          uchar *minIndex) {
  T minValue = data[0];
  uchar minIndexIn = 0;

  for (int i = 1; i < size; ++i) {
    if (data[i] < minValue) {
      minValue = data[i];
      // minValue += data[i];
      minIndexIn = i;
    }
  }

  if (minIndex != nullptr) *minIndex = minIndexIn;

  return minValue;
}

template <typename T>
void FindMin2D(int dir,
               int rows,
               int cols,
               T **data,
               T *result,
               uchar *minIndex) {
  T *vector = new T[max(rows, cols)];
  T minValue = 0;

  if (dir == 0) {  // hor
    for (int lambda = 0; lambda < rows; ++lambda) {
      for (int mu = 0; mu < cols; ++mu)
        vector[mu] = data[lambda][mu];

      if (minIndex != nullptr)
        minValue = FindMin<T>(cols, vector, &minIndex[lambda]);
      else
        minValue = FindMin<T>(cols, vector, nullptr);

      if (result != nullptr)
        result[lambda] = minValue;
    }
  } else {  // ver
    for (int mu = 0; mu < cols; ++mu) {
      for (int lambda = 0; lambda < rows; ++lambda)
        vector[lambda] = data[lambda][mu];

      if (minIndex != nullptr)
        minValue = FindMin<T>(rows, vector, &minIndex[mu]);
      else
        minValue = FindMin<T>(rows, vector, nullptr);

      if (result != nullptr)
        result[mu] = minValue;
    }
  }

  delete [] vector;
}

FILE *CreateLogFile(string filePath) {
  FILE *filePtr = nullptr;
  filePtr = fopen(filePath.c_str(), "wb");
  if (filePtr == nullptr) 
	  printf("Error in creating log file: %s.\n", filePath.c_str());
  return filePtr;  
}

template bool ReadFromFile<float>(const char *fname, float *unary);
template bool ReadFromFile<double>(const char *fname, double *unary);

template uchar** allocate2D<uchar>(int row, int col, uchar value);
template int** allocate2D<int>(int row, int col, int value);
template float** allocate2D<float>(int row, int col, float value);
template double** allocate2D<double>(int row, int col, double value);

template void release2D<uchar>(uchar **data, int row);
template void release2D<int>(int **data, int row);
template void release2D<float>(float **data, int row);
template void release2D<double>(double **data, int row);

template void CopyVector<uchar>(uchar *to, uchar *from, int size);
template void CopyVector<int>(int *to, int *from, int size);
template void CopyVector<float>(float *to, float *from, int size);
template void CopyVector<double>(double *to, double *from, int size);

template void CopyVector2D<uchar>(uchar **to, uchar **from, int offsetVar,
                                  int rowOffset, int row, int col);
template void CopyVector2D<int>(int **to, int **from, int offsetVar,
                                int rowOffset, int row, int col);
template void CopyVector2D<float>(float **to, float **from, int offsetVar,
                                  int rowOffset, int row, int col);
template void CopyVector2D<double>(double **to, double **from, int offsetVar,
                                   int rowOffset, int row, int col);

template int SubtractMin<int>(int *data, int size, uchar *minIndex);
template float SubtractMin<float>(float *data, int size, uchar *minIndex);
template double SubtractMin<double>(double *data, int size, uchar *minIndex);

template void print2D<uchar>(uchar **data, int row, int col, string info);
template void print2D<int>(int **data, int row, int col, string info);
template void print2D<float>(float **data, int row, int col, string info);
template void print2D<double>(double **data, int row, int col, string info);

template int FindMin<int>(int size, int *data, uchar *minIndex);
template float FindMin<float>(int size, float *data, uchar *minIndex);
template double FindMin<double>(int size, double *data, uchar *minIndex);

template void FindMin2D<int>(int dir, int rows, int cols, int **data,
                             int *result, uchar *minIndex);
template void FindMin2D<float>(int dir, int rows, int cols, float **data,
                               float *result, uchar *minIndex);
template void FindMin2D<double>(int dir, int rows, int cols, double **data,
                                double *result, uchar *minIndex);
