# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase


# The table retains its original format for better comparison of parameter settings.
# fmt: off
STRIDED_SLICE_CASE = [
    {"x_shape": [2, 4], "x_dtype": np.float32, "axes": [0, 1], "starts": [1, 0], "strides": [1, 1], "ends": [2, 3]},
    {"x_shape": [2, 4], "x_dtype": np.float32, "axes": [0, 1], "starts": [0, 1], "strides": [2, 2], "ends": [-1, 1000]},
    {"x_shape": [4, 5, 6], "x_dtype": np.float32, "axes": [0, 1, 2], "starts": [-3, 0, 2], "strides": [1, 1, 1], "ends": [3, 2, 4]},
    {"x_shape": [3, 4, 5, 6], "x_dtype": np.float32, "axes": [0, 1, 2], "starts": [0, 1, 2], "strides": [1, 1, 1], "ends": [3, 3, 4]},
    # aten as_stride does not support negative stride
    # {"x_shape": [3, 4, 5, 6], "x_dtype": np.float32, "axes": [0, 1, 2], "starts": [3, 1, 4], "strides": [-1, 1, -1], "ends": [1, 3, 2]},

    {"x_shape": [2, 4], "x_dtype": np.float16, "axes": [0, 1], "starts": [1, 0], "strides": [1, 1], "ends": [2, 3]},
    {"x_shape": [2, 4], "x_dtype": np.float16, "axes": [0, 1], "starts": [0, 1], "strides": [2, 2], "ends": [-1, 1000]},
    {"x_shape": [4, 5, 6], "x_dtype": np.float16, "axes": [0, 1, 2], "starts": [-3, 0, 2], "strides": [1, 1, 1], "ends": [3, 2, 4]},
    {"x_shape": [3, 4, 5, 6], "x_dtype": np.float16, "axes": [0, 1, 2], "starts": [0, 1, 2], "strides": [1, 1, 1], "ends": [3, 3, 4]},
    # {"x_shape": [3, 4, 5, 6], "x_dtype": np.float16, "axes": [0, 1, 2], "starts": [3, 1, 4], "strides": [-1, 1, -1], "ends": [1, 3, 2]},

    {"x_shape": [2, 4], "x_dtype": np.int32, "axes": [0, 1], "starts": [1, 0], "strides": [1, 1], "ends": [2, 3]},
    {"x_shape": [2, 4], "x_dtype": np.int32, "axes": [0, 1], "starts": [0, 1], "strides": [2, 2], "ends": [-1, 1000]},
    {"x_shape": [4, 5, 6], "x_dtype": np.int32, "axes": [0, 1, 2], "starts": [-3, 0, 2], "strides": [1, 1, 1], "ends": [3, 2, 4]},
    {"x_shape": [3, 4, 5, 6], "x_dtype": np.int32, "axes": [0, 1, 2], "starts": [0, 1, 2], "strides": [1, 1, 1], "ends": [3, 3, 4]},
    # {"x_shape": [3, 4, 5, 6], "x_dtype": np.int32, "axes": [0, 1, 2], "starts": [3, 1, 4], "strides": [-1, 1, -1], "ends": [1, 3, 2]},
]
# fmt: on


@ddt
class TestStridedSlice(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 4]
        self.x_dtype = np.float32
        self.axes = [0, 1]
        self.starts = [1, 0]
        self.ends = [2, 3]
        self.strides = [1, 1]

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.strided_slice(x, self.axes, self.starts, self.ends, self.strides)

    def slice_cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        return paddle.strided_slice(x, self.axes, self.starts, self.ends, self.strides)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.slice_cast()
        return out

    @data(*STRIDED_SLICE_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, axes, starts, strides, ends):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.axes = axes
        self.starts = starts
        self.ends = ends
        self.strides = strides
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
