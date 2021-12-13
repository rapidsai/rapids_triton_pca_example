/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <names.h>

#include <cstddef>
#include <rapids_triton/batch/batch.hpp>
#include <rapids_triton/tensor/tensor.hpp>

namespace triton {
namespace backend {
namespace NAMESPACE {

void gpu_infer(const float* X_input, float* X_transformed, const float* mu, const float* components, float* X_workplace,
               std::size_t n_components, std::size_t n_cols, std::size_t n_rows, cudaStream_t stream);
}
}  // namespace backend
}  // namespace triton