#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from tests.op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.base.core as core

paddle.enable_static()

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


class TestNPUReduceProd(OpTest):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"dim": [0]}
        self.outputs = {"Out": self.inputs["X"].prod(axis=tuple(self.attrs["dim"]))}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], ["Out"])

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))

    def init_dtype(self):
        self.dtype = np.float32


class TestNPUReduceProd2(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {}  # default 'dim': [0]
        self.outputs = {"Out": self.inputs["X"].prod(axis=tuple([0]))}


class TestNPUReduceProd3(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        # self.attrs = {'dim': [0]}
        self.outputs = {"Out": self.inputs["X"].prod(axis=tuple([0]))}


class TestNPUReduceProd4(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((32, 8, 50, 2)).astype(self.dtype)}
        self.attrs = {"dim": [-1]}
        self.outputs = {"Out": self.inputs["X"].prod(axis=tuple([-1]))}


class TestNPUReduceProdInt32(TestNPUReduceProd):
    def init_dtype(self):
        self.dtype = np.int32

    # int32 is not supported for gradient check
    def test_check_grad(self):
        pass


class TestNPUReduceProdInt64(TestNPUReduceProd):
    def init_dtype(self):
        self.dtype = np.int64

    # int64 is not supported for gradient check
    def test_check_grad(self):
        pass


class TestNPUReduceProd6D(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 2, 3, 4, 2)).astype(self.dtype)}
        self.attrs = {"dim": [2, 3, 4]}
        self.outputs = {"Out": self.inputs["X"].prod(axis=tuple(self.attrs["dim"]))}


class TestNPUReduceProd8D(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {
            "X": np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype(self.dtype)
        }
        self.attrs = {"dim": [2, 3, 4]}
        self.outputs = {"Out": self.inputs["X"].prod(axis=tuple(self.attrs["dim"]))}


class TestReduceAll(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"reduce_all": True}
        self.outputs = {"Out": self.inputs["X"].prod()}


@skip_check_grad_ci(reason="right now not implement grad op")
class TestNPUReduceProdWithOutDtype_fp16(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"dim": [0], "out_dtype": int(core.VarDesc.VarType.FP16)}
        self.outputs = {
            "Out": self.inputs["X"]
            .prod(axis=tuple(self.attrs["dim"]))
            .astype(np.float16)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    # grad op of fp16 dtype has a very low precision
    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.float16


class TestNPUReduceProdWithOutDtype_fp32(TestNPUReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_npu()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"dim": [0], "out_dtype": int(core.VarDesc.VarType.FP32)}
        self.outputs = {
            "Out": self.inputs["X"]
            .prod(axis=tuple(self.attrs["dim"]))
            .astype(np.float32)
        }


if __name__ == "__main__":
    unittest.main()
