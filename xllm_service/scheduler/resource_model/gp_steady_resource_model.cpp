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

#include "gp_steady_resource_model.h"

#include <algorithm>
#include <cmath>

namespace xllm_service {

GPSteadyResourceModel::GPSteadyResourceModel(
    std::unique_ptr<GaussianProcess> gp_hbm,
    std::unique_ptr<GaussianProcess> gp_compute,
    std::unique_ptr<GaussianProcess> gp_bandwidth)
    : gp_hbm_(std::move(gp_hbm)),
      gp_compute_(std::move(gp_compute)),
      gp_bandwidth_(std::move(gp_bandwidth)) {}

ResourceNeeds GPSteadyResourceModel::calc_3d_resources(
    double token_rate, double moment1, double moment2,
    double decode_pressure) const {
  Eigen::VectorXd x(4);
  x << token_rate, moment1, moment2, decode_pressure;

  ResourceNeeds needs;
  needs.hbm_gb = std::max(gp_hbm_->predict_mean(x), 0.0);
  needs.compute_sm = std::clamp(gp_compute_->predict_mean(x), 0.0, 1.0);
  needs.bandwidth = std::clamp(gp_bandwidth_->predict_mean(x), 0.0, 1.0);
  return needs;
}

ResourceNeeds GPSteadyResourceModel::compute_resource_needs(
    int64_t model_heat) const {
  return calc_3d_resources(static_cast<double>(model_heat), 0.0, 0.0, 0.0);
}

int32_t GPSteadyResourceModel::compute_gpu_target(
    int64_t model_heat, const GpuHardwareSpec& hw) const {
  ResourceNeeds needs = compute_resource_needs(model_heat);

  double hbm_gpus = needs.hbm_gb / hw.hbm_per_gpu_gb;
  double compute_gpus = needs.compute_sm / hw.compute_sm_per_gpu;
  double bandwidth_gpus = needs.bandwidth / hw.bandwidth_per_gpu;

  int32_t target = static_cast<int32_t>(
      std::ceil(std::max({hbm_gpus, compute_gpus, bandwidth_gpus})));
  return std::max(target, 1);
}

std::string GPSteadyResourceModel::name() const { return "gp_steady"; }

}  // namespace xllm_service
