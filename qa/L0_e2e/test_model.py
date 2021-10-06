# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from rapids_triton import Client
from rapids_triton.testing import array_close


@pytest.fixture
def model_inputs():
    # TODO(template): Add data generation/retrieval for model tests

@pytest.fixture
def model_output_sizes():
    # TODO(template): Compute size (in bytes) of outputs and return as
    # dictionary mapping output names to sizes

def get_ground_truth(inputs):
    #TODO(template): Return ground truth expected for given inputs


#TODO(template): Provide names of each model to test
@pytest.mark.parametrize(
    "model_name", ['REPLACE_ME']
)
def test_model(model_name, model_inputs, model_output_sizes):
    client = Client()
    result = client.predict(model_name, model_inputs, model_output_sizes)
    shm_result = client.predict(
        model_name, model_inputs, model_output_sizes, shared_mem='cuda'
    )
    ground_truth = get_ground_truth(model_inputs)

    for output_name in sorted(ground_truth.keys()):
        arrays_close(
            result[output_name],
            ground_truth[output_name],
            atol=1e-5,
            assert_close=True
        )
        arrays_close(
            shm_result[output_name],
            ground_truth[output_name],
            atol=1e-5,
            assert_close=True
        )
