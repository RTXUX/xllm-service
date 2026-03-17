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

#pragma once

#include <Eigen/Dense>
#include <utility>

#include "common/macros.h"

namespace xllm_service {

// Gaussian Process regression with RBF (Squared Exponential) kernel and
// Automatic Relevance Determination (ARD) length scales.
//
// Kernel: k(x, x') = sigma_f^2 * exp(-0.5 * sum_d((x_d - x'_d)^2 / l_d^2))
//
// Training data and hyperparameters are provided at construction time.
// The Cholesky decomposition is pre-computed for O(n) prediction.
class GaussianProcess {
 public:
  // Construct from training data and hyperparameters.
  // X_train:        (N x D) matrix of training inputs
  // y_train:        (N) vector of training outputs
  // lengthscales:   (D) vector of ARD length scales
  // signal_variance: kernel signal variance (sigma_f^2)
  // noise_variance:  observation noise variance (sigma_n^2)
  GaussianProcess(const Eigen::MatrixXd& X_train,
                  const Eigen::VectorXd& y_train,
                  const Eigen::VectorXd& lengthscales,
                  double signal_variance,
                  double noise_variance);

  ~GaussianProcess() = default;

  // Predict mean at a single test point (D-dimensional).
  double predict_mean(const Eigen::VectorXd& x_test) const;

  // Predict mean and variance at a single test point.
  std::pair<double, double> predict(const Eigen::VectorXd& x_test) const;

  int input_dim() const { return input_dim_; }
  int num_train() const { return n_train_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(GaussianProcess);

  // RBF-ARD kernel value between two points.
  double rbf_kernel(const Eigen::VectorXd& x1,
                    const Eigen::VectorXd& x2) const;

  // Kernel vector between a test point and all training points.
  Eigen::VectorXd kernel_vector(const Eigen::VectorXd& x_test) const;

  int input_dim_;
  int n_train_;

  // Hyperparameters
  Eigen::VectorXd lengthscales_;
  double signal_variance_;
  double noise_variance_;

  // Training data
  Eigen::MatrixXd X_train_;  // (N x D)
  Eigen::VectorXd y_train_;  // (N,)

  // Precomputed for fast prediction
  Eigen::LLT<Eigen::MatrixXd> llt_;  // Cholesky of K_train + noise*I
  Eigen::VectorXd alpha_;             // K_train^{-1} * y_train
};

}  // namespace xllm_service
