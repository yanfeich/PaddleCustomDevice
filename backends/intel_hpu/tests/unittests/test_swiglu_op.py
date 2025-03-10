#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.incubate.nn.functional.swiglu as swigluimpl
from tests.op_test import convert_float_to_uint16, convert_uint16_to_float

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


def swiglu_naive(x, y=None):
    if y is None:
        x, y = paddle.chunk(x, chunks=2, axis=-1)
    return F.silu(x) * y


#  只有X，Y为空
class TestSwigluFP32OnlyX(unittest.TestCase):
    def setUp(self):
        self.hpu_place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        self.shape = (20, 512)
        self.init_dtype()

    def init_dtype(self):
        self.dtype = "float32"

    def check_result(self, golden_res, fused_res):
        if self.dtype == "float16":
            rtol = 1e-3
            atol = 1e-3
        elif self.dtype == "bfloat16":
            rtol = 8e-3
            atol = 8e-3
        elif self.dtype == "float32":
            rtol = 1e-5
            atol = 1e-5
        else:
            self.assertTrue(
                False,
                msg="Swiglu input dtype only supports bfloat16, \
                     float16,float32, but got "
                + self.dtype,
            )
        golden_y = golden_res
        fused_y = fused_res
        np.testing.assert_allclose(golden_y, fused_y, rtol=rtol, atol=atol)

    def golden_swiglu(self, x, y=None):
        res = swiglu_naive(x)
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            return res
        return res.numpy()

    def fused_swiglu(self, x, y=None):
        res = swigluimpl(x)
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            return res
        return res.numpy()

    def gen_input(self):
        x = np.random.randn(self.shape[0], self.shape[1])
        return x

    def test_swiglu(self):
        np_x = self.gen_input()
        if self.dtype == "bfloat16":
            np_x = convert_float_to_uint16(np_x)
        golden_x = paddle.to_tensor(
            np_x, place=self.hpu_place, dtype=self.dtype, stop_gradient=False
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.hpu_place, dtype=self.dtype, stop_gradient=False
        )

        golden_res = self.golden_swiglu(golden_x)
        fused_res = self.fused_swiglu(fused_x)
        self.check_result(golden_res, fused_res)


#  X，Y都为空
class TestSwigluFP32BothXY(TestSwigluFP32OnlyX):
    def gen_input(self):
        x = np.random.randn(self.shape[0], self.shape[1])
        y = np.random.randn(self.shape[0], self.shape[1])
        return x, y

    def golden_swiglu(self, x, y=None):
        res = swiglu_naive(x, y)
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            return res
        return res.numpy()

    def fused_swiglu(self, x, y):
        res = swigluimpl(x, y)
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            return res
        return res.numpy()

    def test_swiglu(self):
        np_x, np_y = self.gen_input()
        if self.dtype == "bfloat16":
            np_x = convert_float_to_uint16(np_x)
            np_y = convert_float_to_uint16(np_y)
        golden_x = paddle.to_tensor(
            np_x, place=self.hpu_place, dtype=self.dtype, stop_gradient=False
        )
        golden_y = paddle.to_tensor(
            np_y, place=self.hpu_place, dtype=self.dtype, stop_gradient=False
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.hpu_place, dtype=self.dtype, stop_gradient=False
        )
        fused_y = paddle.to_tensor(
            np_y, place=self.hpu_place, dtype=self.dtype, stop_gradient=False
        )

        golden_res = self.golden_swiglu(golden_x, golden_y)
        fused_res = self.fused_swiglu(fused_x, fused_y)
        self.check_result(golden_res, fused_res)


class TestSwigluBF16OnlyX(TestSwigluFP32OnlyX):
    def init_dtype(self):
        self.dtype = "bfloat16"


class TestSwigluBF16BothXY(TestSwigluFP32BothXY):
    def init_dtype(self):
        self.dtype = "bfloat16"


class TestSwigluFP32OnlyX3D(TestSwigluFP32OnlyX):
    def setUp(self):
        self.hpu_place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        self.shape = (2, 20, 512)
        self.init_dtype()

    def gen_input(self):
        x = np.random.randn(self.shape[0], self.shape[1], self.shape[1])
        return x


class TestSwigluFP32BothXY3D(TestSwigluFP32BothXY):
    def setUp(self):
        self.hpu_place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        self.shape = (2, 20, 512)
        self.init_dtype()

    def gen_input(self):
        x = np.random.randn(self.shape[0], self.shape[1], self.shape[2])
        y = np.random.randn(self.shape[0], self.shape[1], self.shape[2])
        return x, y


class TestSwigluBF16OnlyX3D(TestSwigluFP32OnlyX3D):
    def init_dtype(self):
        self.dtype = "bfloat16"


class TestSwigluBF16BothXY3D(TestSwigluFP32BothXY3D):
    def init_dtype(self):
        self.dtype = "bfloat16"


if __name__ == "__main__":
    np.random.seed(2024)
    unittest.main()
