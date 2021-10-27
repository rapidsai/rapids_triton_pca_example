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
#include <shared_state.h>
#include <gpu_infer.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <rapids_triton/batch/batch.hpp>        // rapids::Batch
#include <rapids_triton/model/model.hpp>        // rapids::Model
#include <rapids_triton/triton/deployment.hpp>  // rapids::DeploymentType
#include <rapids_triton/triton/device.hpp>      // rapids::device_id_t
#include <rapids_triton/triton/logging.hpp>
#include <rapids_triton/memory/buffer.hpp>

namespace triton {
namespace backend {
namespace NAMESPACE {

struct RapidsModel : rapids::Model<RapidsSharedState> {
  RapidsModel(std::shared_ptr<RapidsSharedState> shared_state,
              rapids::device_id_t device_id, cudaStream_t default_stream,
              rapids::DeploymentType deployment_type,
              std::string const& filepath)
      : rapids::Model<RapidsSharedState>(shared_state, device_id,
                                         default_stream, deployment_type,
                                         filepath) {}

  void cpu_infer(const float* X_input, float* X_transformed, const float* mu, const float* components, float* X_workplace,
               std::size_t n_components, std::size_t n_cols, std::size_t n_rows) const
  {
    // Mean center
    for (std::size_t i = 0; i < n_rows; ++i) {
      for (std::size_t j = 0; j < n_cols; ++j) {
        X_workplace[i * n_cols + j] = X_input[i * n_cols + j] - mu[j];
      }
    }

    // Dot product
    for (std::size_t i = 0; i < n_rows; i++)
      for (std::size_t j = 0; j < n_cols; j++)
        for (std::size_t k = 0; k < n_components; k++)
          X_transformed[i * n_components + k] += \
           X_workplace[i * n_cols + j] * components[j * n_components + k];
  }

  void predict(rapids::Batch& batch) const {
    auto X_input = get_input<float>(batch, "X_input");
    auto X_transformed = get_output<float>(batch, "X_transformed");
    auto n_components = get_shared_state()->n_components;
    auto n_cols = get_shared_state()->n_cols;
    auto n_rows = X_input.shape()[0];
    auto memory_type = X_input.mem_type();

    auto X_workplace = rapids::Buffer<float>(n_cols * n_rows, memory_type, get_device_id(), get_stream());
    if (memory_type == rapids::DeviceMemory) {
      gpu_infer(X_input.data(), X_transformed.data(), mu.data(), components.data(), X_workplace.data(),
                n_components, n_cols, n_rows, get_stream());
      rapids::detail::copy( 
        X_transformed.data(), X_workplace.data(),
        X_transformed.size(), get_stream(), memory_type, memory_type);
    }
    else {
      cpu_infer(X_input.data(), X_transformed.data(), mu.data(), components.data(), X_workplace.data(),
                n_components, n_cols, n_rows);
    }

    X_transformed.finalize();

  }

  auto load_file(const std::string& file_path, std::size_t expected_size, const rapids::MemoryType& memory_type) {
    std::ifstream data_file(file_path, std::ios::binary);
    std::vector<unsigned char> data_vector(std::istreambuf_iterator<char>(data_file), {});
    if (data_vector.size() != expected_size) {
      throw "Invalid size. Expected " + std::to_string(expected_size) + " but got " + std::to_string(data_vector.size());
    }
    auto result = rapids::Buffer<float>(data_vector.size() / sizeof (float), memory_type, get_device_id());
    rapids::copy(result, rapids::Buffer<float>(reinterpret_cast<float*>(data_vector.data()),
                                               data_vector.size() / sizeof (float),
                                               rapids::HostMemory));
    return result;
  }

  void load() {
    rapids::log_info(__FILE__, __LINE__) << "Starting loading ...";
    auto n_components = get_shared_state()->n_components;
    auto n_cols = get_shared_state()->n_cols;
    auto memory_type = rapids::MemoryType{};
    if (get_deployment_type() == rapids::GPUDeployment) {
      memory_type = rapids::DeviceMemory;
    } else {
      memory_type = rapids::HostMemory;
    }

    auto path = std::filesystem::path(get_filepath());
    /* If the config file does not specify a filepath for the model,
     * get_filepath returns the directory where the serialized model should be
     * found. It is generally good practice to provide logic to allow the use
     * of a default filename so that model configurations do not always have to
     * specify a path to their model */
    if (!std::filesystem::is_directory(path)) {
      throw std::exception();
    }
    rapids::log_info(__FILE__, __LINE__) << "Loading components vector";
    components = load_file(get_filepath() + "/components.bin",
                           n_components * n_cols * sizeof(float),
                           memory_type);
    rapids::log_info(__FILE__, __LINE__) << "Loading mu vector";
    mu = load_file(get_filepath() + "/mu.bin",
                   n_cols * sizeof(float),
                   memory_type);
  }
  void unload() {}

  private:
    rapids::Buffer<float> components{};
    rapids::Buffer<float> mu{};
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
