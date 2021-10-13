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
#include <rapids_triton/memory/resource.hpp>

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

  void predict(rapids::Batch& batch) const {
    auto X_input = get_input<float>(batch, "X_input");
    auto n_components = get_shared_state()->n_components;
    auto n_cols = get_shared_state()->n_cols;
    auto n_rows = X_input.size();
    auto X_transformed = get_output<float>(batch, "X_transformed");
    auto memory_type = rapids::MemoryType{};
    if constexpr (rapids::IS_GPU_BUILD) {
      if (get_deployment_type() == rapids::GPUDeployment) {
        memory_type = rapids::DeviceMemory;
        rapids::cuda_check(cudaSetDevice(get_device_id()));
      } else {
        memory_type = rapids::HostMemory;
      }
    } else {
      memory_type = rapids::HostMemory;
    }

    auto X_workplace = rapids::Buffer<float>(n_cols * n_rows, memory_type, get_device_id(), get_stream());

    gpu_infer(X_input.data(), X_transformed.data(), mu.data(), components.data(), X_workplace.data(),
              n_components, n_cols, n_rows, get_stream());
    X_transformed.finalize();
  }

  auto load_file(const std::string& file_path, std::size_t expected_size, const rapids::MemoryType& memory_type) {
    std::ifstream data_file(file_path, std::ios::binary);
    std::vector<unsigned char> data_vector(std::istreambuf_iterator<char>(data_file), {});
    if (data_vector.size() != expected_size) {
      throw "Invalid size. Expected " + std::to_string(expected_size) + " but got " + std::to_string(data_vector.size());
    }
    auto result = rapids::Buffer<float>(data_vector.size(), memory_type, get_device_id());
    rapids::copy(result, rapids::Buffer<float>(reinterpret_cast<float*>(data_vector.data()),
                                               data_vector.size(),
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
    components = std::move(load_file(get_filepath() + "/components.bin",
                           n_components * n_cols * sizeof(float),
                           memory_type));
    rapids::log_info(__FILE__, __LINE__) << "Loading mu vector";
    mu = std::move(load_file(get_filepath() + "/mu.bin",
                   n_cols * sizeof(float),
                   memory_type));
  }
  void unload() {}

  private:
    rapids::Buffer<float> components{};
    rapids::Buffer<float> mu{};
};

}  // namespace NAMESPACE
}  // namespace backend
}  // namespace triton
