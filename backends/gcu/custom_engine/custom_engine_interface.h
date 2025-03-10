// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/fluid/custom_engine/custom_engine_ext.h"
#include "paddle/phi/extension.h"

#ifdef __cplusplus
extern "C" {
#endif

C_Status RegisterCustomEngineOp();
C_Status CustomEngineOpLower(C_CustomEngineLowerParams* lower_param);
C_Status GraphEngineBuild(C_CustomEngineInstruction instruction);
C_Status GraphEngineExecute(C_CustomEngineInstruction instruction);

void InitPluginCustomEngine(CustomEngineParams* params);

#ifdef __cplusplus
} /* extern "c" */
#endif
