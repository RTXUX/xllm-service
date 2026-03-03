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

#include "common/macros.h"
#include "resource_model.h"

namespace xllm_service {

class LinearResourceModel final : public ResourceModel {
 public:
  LinearResourceModel(double hbm_a, double hbm_b,
                      double compute_a, double compute_b);
  ~LinearResourceModel() override = default;

  int32_t compute_gpu_target(int64_t model_heat,
                             const GpuHardwareSpec& hw) const override;
  std::string name() const override;

 private:
  DISALLOW_COPY_AND_ASSIGN(LinearResourceModel);
  double hbm_a_, hbm_b_;
  double compute_a_, compute_b_;
};

}  // namespace xllm_service
