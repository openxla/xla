/* Copyright 2025 The OpenXLA Authors.

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

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_EXTRACTTMAINFOPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

struct ExtractTmaInfoPass
    : public impl::ExtractTmaInfoPassBase<ExtractTmaInfoPass> {
  std::optional<SwizzleMode> GetSwizzleMode(Location loc, TensorDescType type) {
    auto swizzle_mode = nvidia_gpu::getTMASwizzleMode(loc, type);
    if (failed(swizzle_mode)) {
      return std::nullopt;
    }
    SwizzleMode swizzle_mode_enum;
    switch (swizzle_mode.value()) {
      case 0:
        swizzle_mode_enum = SwizzleMode::kNone;
        break;
      case 1:
        swizzle_mode_enum = SwizzleMode::k32b;
        break;
      case 2:
        swizzle_mode_enum = SwizzleMode::k64b;
        break;
      case 3:
        swizzle_mode_enum = SwizzleMode::k128b;
        break;
      default:
        return std::nullopt;
    }
    return swizzle_mode_enum;
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mod.walk([&](mlir::triton::FuncOp func_op) -> void {
      for (auto arg : func_op.getArguments()) {
        if (!isa<TensorDescType>(arg.getType())) {
          continue;
        }

        auto tma_descriptor_attr = func_op.getArgAttrOfType<TmaDescriptorAttr>(
            arg.getArgNumber(), "tt.tma_descriptor");
        if (!tma_descriptor_attr) {
          emitError(arg.getLoc(),
                    "Argument of type tt.tensordesc must have attribute "
                    "tt.tma_descriptor");
          signalPassFailure();
          return;
        }

        auto tensor_desc_type = mlir::cast<TensorDescType>(arg.getType());
        auto swizzle_mode = GetSwizzleMode(arg.getLoc(), tensor_desc_type);
        if (!swizzle_mode.has_value()) {
          emitError(arg.getLoc(),
                    "Unable to determine swizzle mode from TensorDescType");
          signalPassFailure();
          return;
        }

        // When `num_ctas > 1`, Triton's CGA layout assignment can split a
        // tensor descriptor's shared-memory layout across the CTAs of a
        // cluster, so that each CTA's TMA load only fills its own slice.
        // Triton sizes device-side boxes the same way (`getShapePerCTA` in
        // `createTMADesc`, TMAUtilities.cpp). `tt.tma_descriptor` still
        // carries the cluster-wide tile shape, so divide each split dimension
        // of `tile_shape` by its CTA split factor. Otherwise, the host TMA
        // `box_dims` would overflow each CTA's shared-memory slice and its
        // mbarrier transaction byte count. Unsplit dimensions (including all
        // dimensions when `num_ctas == 1`) are left untouched.
        llvm::SmallVector<int64_t> tile_shape(
            tma_descriptor_attr.getTileShape());
        if (mlir::Attribute shared_layout =
                tensor_desc_type.getSharedLayout()) {
          // Both `desc_shape` and `cta_shape` are ordered major-to-minor,
          // while `layout` lists the dimensions of `tile_shape` minor-to-major.
          llvm::ArrayRef<int64_t> desc_shape = tensor_desc_type.getShape();
          llvm::SmallVector<int64_t> cta_shape =
              mlir::triton::gpu::getShapePerCTA(shared_layout, desc_shape);
          llvm::ArrayRef<int64_t> layout = tma_descriptor_attr.getLayout();
          const int64_t rank = desc_shape.size();
          for (int64_t desc_dim = 0; desc_dim < rank; ++desc_dim) {
            if (cta_shape[desc_dim] <= 0 ||
                desc_shape[desc_dim] == cta_shape[desc_dim]) {
              continue;
            }
            const int64_t split = desc_shape[desc_dim] / cta_shape[desc_dim];
            const int64_t tile_dim = static_cast<int64_t>(layout.size()) == rank
                                         ? layout[rank - 1 - desc_dim]
                                         : -1;
            if (tile_dim < 0 ||
                desc_shape[desc_dim] % cta_shape[desc_dim] != 0 ||
                tile_shape[tile_dim] % split != 0) {
              emitError(arg.getLoc(),
                        "Unable to split the TMA tile shape across CTAs: "
                        "cluster tile dimension is not divisible by the CTA "
                        "split factor");
              signalPassFailure();
              return;
            }
            tile_shape[tile_dim] /= split;
          }
        }

        IRRewriter rewriter(&getContext());
        func_op.setArgAttr(
            arg.getArgNumber(), "tt.tma_descriptor",
            rewriter.getAttr<TmaDescriptorAttr>(
                tma_descriptor_attr.getGlobalShape(), tile_shape,
                tma_descriptor_attr.getTileStrides(),
                tma_descriptor_attr.getLayout(),
                tma_descriptor_attr.getElementByteSize(),
                SwizzleModeAttr::get(&getContext(), swizzle_mode.value())));
      }
    });
  }
};

}  // namespace

}  // namespace mlir::triton::xla
