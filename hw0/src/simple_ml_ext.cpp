#include <cmath>
#include <cstddef>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

static void matmul(float *C, const float *A, const float *B, size_t row,
                   size_t col, size_t reduce) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < reduce; ++k) {
        sum += A[i * reduce + k] * B[k * col + j];
      }
      C[i * col + j] = sum;
    }
  }
}

static void transpose(float *dst, const float *src, size_t row, size_t col) {
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      dst[i * col + j] = src[j * row + i];
    }
  }
}

static void computeGradient(std::vector<float> &gradient,
                            const std::vector<float> &XTheta, const float *X,
                            const unsigned char *y, const size_t batch_size,
                            const size_t input_dim, const size_t num_classes) {
  std::vector<float> expXTheta(XTheta.size(), 0.0f);
  std::vector<float> sumExp(batch_size, 0.0f);

  // 1. compute expXTheta
  for (size_t i = 0; i < batch_size; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < num_classes; ++j) {
      float _exp = std::exp(XTheta[i * num_classes + j]);
      sum += _exp;
      expXTheta[i * num_classes + j] = _exp;
    }
    sumExp[i] = sum;
  }

  // 2. compute Z
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < num_classes; ++j) {
      expXTheta[i * num_classes + j] /= sumExp[i];
    }
  }

  // 3. Z - Iy
  for (size_t i = 0; i < batch_size; ++i) {
    expXTheta[i * num_classes + y[i]] -= 1.0f;
  }

  // 4. do transpose and matmul
  std::vector<float> transposeX(input_dim * batch_size, 0.0f);
  transpose(transposeX.data(), X, input_dim, batch_size);
  matmul(gradient.data(), transposeX.data(), expXTheta.data(), input_dim,
         num_classes, batch_size);
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */

  /// BEGIN YOUR CODE
  size_t num_iters = (m + batch - 1) / batch;
  for (size_t i = 0; i < num_iters; ++i) {
    size_t start = i * batch;
    size_t end = std::min(m, (i + 1) * batch);
    size_t batch_size = end - start;
    std::vector<float> XTheta(batch_size * k, 0.0f);
    matmul(XTheta.data(), X + start * n, theta, batch_size, k, n);

    std::vector<float> gradient(n * k, 0.0f);
    computeGradient(gradient, XTheta, X + start * n, y + start, batch_size, n,
                    k);
    // update theta
    float _lr = lr / batch_size;
    for (size_t ii = 0; ii < n; ++ii) {
      for (size_t j = 0; j < k; ++j) {
        theta[ii * k + j] -= _lr * gradient[ii * k + j];
      }
    }
  }
  /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float *>(X.request().ptr),
            static_cast<const unsigned char *>(y.request().ptr),
            static_cast<float *>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
