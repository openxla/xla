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

#ifndef XLA_CODEGEN_INTRINSIC_CPP_LOG_H_
#define XLA_CODEGEN_INTRINSIC_CPP_LOG_H_

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

double log_f64(double x) asm("xla.log.f64");
Vec2d log_v2f64(Vec2d x) asm("xla.log.v2f64");
Vec4d log_v4f64(Vec4d x) asm("xla.log.v4f64");
Vec8d log_v8f64(Vec8d x) asm("xla.log.v8f64");

double log2_f64(double x) asm("xla.log2.f64");
Vec2d log2_v2f64(Vec2d x) asm("xla.log2.v2f64");
Vec4d log2_v4f64(Vec4d x) asm("xla.log2.v4f64");
Vec8d log2_v8f64(Vec8d x) asm("xla.log2.v8f64");

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_INTRINSIC_CPP_LOG_H_
