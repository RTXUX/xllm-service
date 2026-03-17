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

#include <cstdint>
#include <string>

#include "common/types.h"

namespace xllm_service {

class ResourceModel {
 public:
  virtual ~ResourceModel() = default;
  virtual int32_t compute_gpu_target(int64_t model_heat,
                                     const GpuHardwareSpec& hw) const = 0;
  virtual ResourceNeeds compute_resource_needs(int64_t model_heat) const = 0;
  virtual std::string name() const = 0;

  // GP steady pool interface: compute 3D resource needs from traffic stats.
  // Default implementation delegates to compute_resource_needs(model_heat).
  virtual ResourceNeeds calc_3d_resources(double token_rate,
                                          double moment1,
                                          double moment2,
                                          double decode_pressure) const {
    return compute_resource_needs(static_cast<int64_t>(token_rate));
  }

  // GP dynamic pool interface: compute required instance count for target SLO.
  // Default implementation delegates to compute_gpu_target(model_heat, hw).
  virtual int32_t calc_instance_number(double token_rate,
                                       double moment1,
                                       double moment2,
                                       double target_slo_rate) const {
    return compute_gpu_target(static_cast<int64_t>(token_rate),
                              GpuHardwareSpec{});
  }
};

}  // namespace xllm_service
