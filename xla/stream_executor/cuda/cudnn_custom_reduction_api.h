/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDNN_CUSTOM_REDUCTION_API_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDNN_CUSTOM_REDUCTION_API_H_

#include "third_party/cudnn_frontend/include/cudnn_frontend_version.h"
#include "third_party/gpus/cudnn/cudnn_version.h"

// XLA_HAS_CUDNN_CUSTOM_REDUCTION_API is 1 when the build can use the cuDNN
// frontend custom reduction API, and 0 otherwise.
// - The frontend has the API (cudnn_frontend/node/custom_reduction.h) since
//   release 1.33.0.
// - The frontend compiles its custom reduction backend code only with cuDNN
//   9.29+ headers: with older headers, CustomReductionNode::expand_node always
//   returns GRAPH_NOT_SUPPORTED, so the build cannot use the API.
#if CUDNN_FRONTEND_VERSION >= 13300 && CUDNN_VERSION >= 92900
#define XLA_HAS_CUDNN_CUSTOM_REDUCTION_API 1
#else
#define XLA_HAS_CUDNN_CUSTOM_REDUCTION_API 0
#endif

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDNN_CUSTOM_REDUCTION_API_H_
