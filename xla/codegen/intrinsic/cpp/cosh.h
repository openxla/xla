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

#ifndef XLA_CODEGEN_INTRINSIC_CPP_COSH_H_
#define XLA_CODEGEN_INTRINSIC_CPP_COSH_H_

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

float cosh_f32(float x) asm("xla.cosh.f32");
Vec4f cosh_v4f32(Vec4f x) asm("xla.cosh.v4f32");
Vec8f cosh_v8f32(Vec8f x) asm("xla.cosh.v8f32");
Vec16f cosh_v16f32(Vec16f x) asm("xla.cosh.v16f32");

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_INTRINSIC_CPP_COSH_H_
