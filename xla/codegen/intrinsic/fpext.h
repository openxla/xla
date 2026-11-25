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

#ifndef XLA_CODEGEN_INTRINSIC_FPEXT_H_
#define XLA_CODEGEN_INTRINSIC_FPEXT_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

// XLA intrinsic for extending 8-bit floating-point values (scalars and
// vectors) to wider floating-point types.
class FpExt : public Intrinsic<FpExt> {
 public:
  static constexpr absl::string_view kName = "fpext";
  static constexpr bool kLastArgIsReturnType = true;
  static constexpr int8_t kNumArgs = 2;  // Second arg is the return type.
  static std::vector<std::vector<Type>> SupportedVectorTypes(
      absl::string_view features = "") {
    std::vector<std::vector<Type>> supported_types;
    for (PrimitiveType f8_type : {F8E4M3, F8E4M3FN, F8E4M3B11FNUZ, F8E3M4,
                                  F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F8E8M0FNU}) {
      for (PrimitiveType fx_type : {F16, BF16, F32, F64}) {
        supported_types.push_back({Type::S(f8_type), Type::S(fx_type)});
        for (int8_t vector_width : {2, 4, 8, 16}) {
          supported_types.push_back(
              {Type::V(f8_type, vector_width), Type::V(fx_type, vector_width)});
        }
      }
    }
    return supported_types;
  }

  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          Type from, Type to);
};

}  // namespace xla::codegen::intrinsics

#endif  // XLA_CODEGEN_INTRINSIC_FPEXT_H_
