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

#include <names.h>
#include <shared_state.h>
#include <gpu_infer.h>

#include <cstddef>
#include <raft/handle.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/transpose.h>
#include <raft/stats/mean_center.cuh>
#include <rapids_triton/batch/batch.hpp>
#include <rapids_triton/tensor/tensor.hpp>

namespace triton { namespace backend { namespace NAMESPACE {
    void gpu_infer(const float* X_input, float* X_transformed, const float* mu, const float* components, float* X_workplace,
                   std::size_t n_components, std::size_t n_cols, std::size_t n_rows, cudaStream_t stream) {
        raft::stats::meanCenter(X_workplace, X_input, mu, n_cols, n_rows, true, true, stream);
        float alpha = 1;
        float beta  = 0;
        auto handle = raft::handle_t(1);

        handle.set_stream(stream);
        raft::linalg::gemm(handle,
                           X_workplace,
                           static_cast<int>(n_cols),
                           static_cast<int>(n_rows),
                           components,
                           X_transformed,
                           static_cast<int>(n_rows),
                           static_cast<int>(n_components),
                           CUBLAS_OP_T,
                           CUBLAS_OP_T,
                           alpha,
                           beta,
                           stream);
        raft::linalg::transpose(handle, X_transformed, X_workplace,
            static_cast<int>(n_rows), static_cast<int>(n_components), stream);
    }

}}}