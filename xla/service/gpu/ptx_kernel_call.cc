/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/ptx_kernel_call.h"
#include "xla/tsl/platform/logging.h"

#include <cstdint>
#include <utility>

#include "absl/strings/string_view.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

namespace xla::gpu {

PtxCall PtxCall::Parse(absl::string_view backend_config,
                       mlir::MLIRContext* mlir_context) {
  auto attrs = mlir::cast<mlir::DictionaryAttr>(
      mlir::parseAttribute(backend_config, mlir_context));
  auto name = attrs.getAs<mlir::StringAttr>("name").getValue().str();
  auto source = attrs.getAs<mlir::StringAttr>("source").str();
  VLOG(2) << "Dumping all attributes in backend_config:";
  for (const auto& namedAttr : attrs) {
    std::string value_str;
    llvm::raw_string_ostream os(value_str);
    namedAttr.getValue().print(os);
    VLOG(2) << "  " << namedAttr.getName().str() << ": " << value_str;
  }

  auto get_int32_attr = [&attrs](const char* attr_name) -> int32_t {
    return static_cast<int32_t>(
        attrs.getAs<mlir::IntegerAttr>(attr_name).getValue().getSExtValue());
  };
  int32_t grid_x = get_int32_attr("grid_x");
  int32_t grid_y = get_int32_attr("grid_y");
  int32_t grid_z = get_int32_attr("grid_z");
  int32_t block_x = get_int32_attr("block_x");
  int32_t block_y = get_int32_attr("block_y");
  int32_t block_z = get_int32_attr("block_z");
  int32_t shared_mem = get_int32_attr("shared_mem_bytes");
  mlir::ArrayAttr output_indices =
      attrs.getAs<mlir::ArrayAttr>("output_indices");
  std::vector<int32_t> output_indices_vec;
  if (output_indices) {
    for (const mlir::Attribute& index : output_indices) {
      output_indices_vec.push_back(
          mlir::cast<mlir::IntegerAttr>(index).getValue().getSExtValue());
    }
  }
  stream_executor::BlockDim block_dim(grid_x, grid_y, grid_z);
  stream_executor::ThreadDim thread_dim(block_x, block_y, block_z);

  return PtxCall{std::move(name),
                 std::move(source),
                 block_dim,
                 thread_dim,
                 static_cast<size_t>(shared_mem),
                 output_indices_vec};
}

}  // namespace xla::gpu
