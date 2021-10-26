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
from rapids_triton.testing import arrays_close, get_random_seed


N_ROWS = 8192
N_COLS = 25
N_COMPONENTS = 5

@pytest.fixture
def model_inputs():
    np.random.seed(get_random_seed())
    return {
        "X_input": np.random.rand(N_ROWS, N_COLS).astype('float32')
    }

@pytest.fixture
def model_output_sizes():
    return {"X_transformed": N_ROWS * N_COMPONENTS * np.dtype('float32').itemsize}

def get_ground_truth(inputs):
    x = inputs['X_input']
    x_c = x - MU
    return {'X_transformed': x_c.dot(COMPONENTS)}


@pytest.mark.parametrize(
    "model_name", ['pca_example']
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

MU = np.array([0.3054301 , 0.53497523, 0.02903529, 0.23445411, 0.41508475,
       0.73335785, 0.89488304, 0.31067532, 0.9334298 , 0.02269967,
       0.75677216, 0.32904336, 0.63879555, 0.75856906, 0.93770117,
       0.80694044, 0.14879903, 0.8788233 , 0.36914352, 0.89124376,
       0.76835155, 0.01684399, 0.1580411 , 0.35072792, 0.38621086],
      dtype=np.float32)
COMPONENTS = np.array([[0.1077859 , 0.0152536 , 0.14996086, 0.27519643, 0.5466197 ],
       [0.47137365, 0.7524288 , 0.16581082, 0.6583814 , 0.6733525 ],
       [0.5419624 , 0.53981566, 0.4943707 , 0.60533386, 0.9173961 ],
       [0.49503392, 0.4416264 , 0.12268677, 0.26787782, 0.910786  ],
       [0.71058154, 0.3931972 , 0.78567946, 0.8114448 , 0.28378612],
       [0.76400083, 0.710263  , 0.9714428 , 0.59266746, 0.63176847],
       [0.47967914, 0.7907602 , 0.14844431, 0.17678756, 0.9410757 ],
       [0.13820966, 0.3714162 , 0.19777128, 0.9384368 , 0.69669586],
       [0.46815118, 0.20329583, 0.3123208 , 0.6186174 , 0.2085056 ],
       [0.4300877 , 0.84767324, 0.42783308, 0.1778231 , 0.3636397 ],
       [0.1769452 , 0.5860459 , 0.37256172, 0.71824384, 0.9448562 ],
       [0.49792168, 0.42727843, 0.8448393 , 0.77229506, 0.09547652],
       [0.33963397, 0.85927695, 0.31496638, 0.35328254, 0.10459802],
       [0.39113268, 0.91155696, 0.73254997, 0.26312187, 0.777164  ],
       [0.07265835, 0.09515466, 0.13576192, 0.26306516, 0.38162884],
       [0.8208812 , 0.33372718, 0.6603761 , 0.14251982, 0.63563746],
       [0.6512604 , 0.41092023, 0.7265426 , 0.9646286 , 0.21258278],
       [0.4980957 , 0.38877907, 0.8429187 , 0.09256837, 0.811749  ],
       [0.13165434, 0.22899932, 0.50088805, 0.9763909 , 0.50195044],
       [0.9490048 , 0.60583454, 0.03239321, 0.04777756, 0.51496094],
       [0.6111744 , 0.35173875, 0.6366924 , 0.56868726, 0.6552913 ],
       [0.41361338, 0.59937996, 0.41819212, 0.52223563, 0.6873631 ],
       [0.07992661, 0.5735988 , 0.49894568, 0.07927666, 0.5696119 ],
       [0.7249317 , 0.25087562, 0.42774037, 0.2647722 , 0.5418794 ],
       [0.19648804, 0.9403854 , 0.25328928, 0.76671   , 0.5263434 ]],
      dtype=np.float32)
