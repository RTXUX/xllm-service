/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "gaussian_process.h"

#include <cmath>
#include <glog/logging.h>

namespace xllm_service {

GaussianProcess::GaussianProcess(const Eigen::MatrixXd& X_train,
                                 const Eigen::VectorXd& y_train,
                                 const Eigen::VectorXd& lengthscales,
                                 double signal_variance,
                                 double noise_variance)
    : input_dim_(X_train.cols()),
      n_train_(X_train.rows()),
      lengthscales_(lengthscales),
      signal_variance_(signal_variance),
      noise_variance_(noise_variance),
      X_train_(X_train),
      y_train_(y_train) {
  CHECK_EQ(X_train.rows(), y_train.size())
      << "X_train rows must match y_train size";
  CHECK_EQ(lengthscales.size(), input_dim_)
      << "Lengthscales dimension must match input dimension";
  CHECK_GT(signal_variance, 0.0) << "Signal variance must be positive";
  CHECK_GE(noise_variance, 0.0) << "Noise variance must be non-negative";

  // Build kernel matrix K_train (N x N)
  Eigen::MatrixXd K(n_train_, n_train_);
  for (int i = 0; i < n_train_; ++i) {
    for (int j = i; j < n_train_; ++j) {
      double k_val = rbf_kernel(X_train_.row(i), X_train_.row(j));
      K(i, j) = k_val;
      K(j, i) = k_val;
    }
    // Add noise variance to diagonal
    K(i, i) += noise_variance_;
  }

  // Cholesky decomposition: K = L * L^T
  llt_.compute(K);
  if (llt_.info() != Eigen::Success) {
    // Add jitter for numerical stability and retry
    LOG(WARNING) << "GP Cholesky failed, adding jitter (1e-6) to diagonal";
    K.diagonal().array() += 1e-6;
    llt_.compute(K);
    CHECK(llt_.info() == Eigen::Success)
        << "GP Cholesky decomposition failed even with jitter";
  }

  // Precompute alpha = K^{-1} * y_train via Cholesky solve
  alpha_ = llt_.solve(y_train_);

  LOG(INFO) << "GaussianProcess initialized: " << n_train_ << " training points"
            << ", input_dim=" << input_dim_
            << ", signal_var=" << signal_variance_
            << ", noise_var=" << noise_variance_;
}

double GaussianProcess::rbf_kernel(const Eigen::VectorXd& x1,
                                   const Eigen::VectorXd& x2) const {
  // k(x1, x2) = sigma_f^2 * exp(-0.5 * sum_d((x1_d - x2_d)^2 / l_d^2))
  Eigen::VectorXd diff = (x1 - x2).array() / lengthscales_.array();
  return signal_variance_ * std::exp(-0.5 * diff.squaredNorm());
}

Eigen::VectorXd GaussianProcess::kernel_vector(
    const Eigen::VectorXd& x_test) const {
  Eigen::VectorXd k_star(n_train_);
  for (int i = 0; i < n_train_; ++i) {
    k_star(i) = rbf_kernel(x_test, X_train_.row(i));
  }
  return k_star;
}

double GaussianProcess::predict_mean(const Eigen::VectorXd& x_test) const {
  Eigen::VectorXd k_star = kernel_vector(x_test);
  return k_star.dot(alpha_);
}

std::pair<double, double> GaussianProcess::predict(
    const Eigen::VectorXd& x_test) const {
  Eigen::VectorXd k_star = kernel_vector(x_test);

  // Mean: k*^T * alpha
  double mean = k_star.dot(alpha_);

  // Variance: k(x*, x*) - k*^T * K^{-1} * k*
  // Using Cholesky: v = L \ k*, variance = k** - v^T * v
  Eigen::VectorXd v = llt_.matrixL().solve(k_star);
  double k_ss = signal_variance_;  // k(x*, x*) = sigma_f^2 (self-kernel)
  double variance = k_ss - v.squaredNorm();

  // Clamp variance to non-negative (numerical stability)
  variance = std::max(variance, 0.0);

  return {mean, variance};
}

}  // namespace xllm_service
