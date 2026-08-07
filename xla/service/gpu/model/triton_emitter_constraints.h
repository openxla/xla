/* Copyright 2024 The OpenXLA Authors.

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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"
#include "xla/codegen/xtile/codegen/tiled_emitter_constraints.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/triton_temporary_memory_estimator.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"

#ifndef XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_
#define XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_

namespace xla {
namespace gpu {

// Triton-specific constraints on tile sizes.
class TritonEmitterConstraints : public EmitterSpecificConstraints {
 public:
  static EmitterSpecificConstraintsBuilder GetBuilder(
      const se::DeviceDescription& device_description);

  absl::StatusOr<bool> ParametersSatisfyConstraints(
      absl::Span<const int64_t> tile_parameters) const override;

  bool HasCustomConstraints() const { return !custom_constraints_.empty(); }

 private:
  // Holds a constraint expression over derived parameters (d'0, ..., d'm) where
  //   (d'0, ..., d'm) = tile_parameters_transform(tile_parameters).
  struct CustomConstraints {
    SymbolicMap tile_parameters_transform;
    ConstraintExpression constraints;
  };

  // Holds the info needed to validate whether the tiling parameters satisfy the
  // constraint that they are either powers of 2, or equal to the dimension
  // size.
  struct RootTileInfo {
    SymbolicMap size_map;
    std::vector<int64_t> dim_sizes;
  };

  // Holds the info needed to estimate the shared memory required by a
  // shared-memory-staging instruction (see `triton_shared_memory_estimator.h`
  // for how staging ops are classified and sized). The size maps are evaluated
  // at concrete tile parameters in `ParametersSatisfyConstraints` to build the
  // `SmemStagingOp` descriptors passed to the shared `EstimateBlockSmemBytes`.
  struct StagingTileInfo {
    // Name of the staging instruction (for diagnostics).
    std::string name;
    // How the tile is staged in shared memory.
    SmemStagingKind kind;
    // Byte size of the element type of the staging instruction.
    int64_t element_byte_size;
    // Size map of the staging instruction's own tile (used for kLayout).
    SymbolicMap size_map;
    // For kDot: size maps of the operand tiles whose K-tile staging buffers are
    // summed to size the shared-memory operand staging.
    llvm::SmallVector<SymbolicMap, 2> operand_size_maps;
  };

  explicit TritonEmitterConstraints(
      llvm::SmallVector<SymbolicMap, 4> tile_size_maps,
      llvm::SmallVector<StagingTileInfo, 2> staging_infos,
      llvm::SmallVector<RootTileInfo, 2> roots,
      std::vector<CustomConstraints> custom_constraints,
      const Shape& root_shape, const se::DeviceDescription& device_info,
      std::unique_ptr<TiledEmitterConstraints> tiled_emitter_constraints)
      : tile_size_maps_(std::move(tile_size_maps)),
        staging_infos_(std::move(staging_infos)),
        roots_(std::move(roots)),
        custom_constraints_(std::move(custom_constraints)),
        root_shape_(root_shape),
        device_info_(device_info),
        tiled_emitter_constraints_(std::move(tiled_emitter_constraints)) {}

  // Derives a vector of `CustomConstraints` to be checked within
  // `ParametersSatisfyConstraints` from a vector of
  // `SymbolicTiledHloInstruction`s representing a symbolically tiled HLO
  // computation. The fusion adaptor is used to figure out which instructions
  // within the computation are operands of the fusion.
  //
  // Currently, this is used to work around an issue with reshapes/bitcasts when
  // instructions are tiled with non-power-of-2 shapes. The resulting custom
  // constraints contain
  //   * the reshape/bitcast's tile size map; this to allow deriving the
  //     output tile sizes for the reshape/bitcast instruction;
  //   * the constraint expression corresponding to the SymbolicTile derived
  //     from the reshape/bitcast instruction's output-to-input indexing map
  //     "in a vacuum" (i.e., without composing with any other indexing map).
  //
  // TODO(b/365727080): move tile derivation to have powers of 2 tiles
  // everywhere, and deprecate this.
  static std::vector<CustomConstraints> DeriveCustomConstraints(
      const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
          instructions,
      const HloFusionAdaptor& fusion_adaptor);

  // A collection of unique size maps from all the
  // `SymbolicTiledHloInstruction`s.
  //
  // Different `TiledHloInstruction`s often have the same size map, so we keep a
  // collection of unique maps to improve compilation time.
  llvm::SmallVector<SymbolicMap, 4> tile_size_maps_;

  // Holds the info needed to estimate the shared memory required by each
  // shared-memory-staging instruction, used to check the shared-memory
  // constraint in `ParametersSatisfyConstraints`.
  llvm::SmallVector<StagingTileInfo, 2> staging_infos_;

  // Holds the info for all fusion roots necessary to check whether the tile
  // sizes evaluate to powers of 2 or have the same size as the dimension, and
  // to estimate the shared memory required to stage the root tiles.
  llvm::SmallVector<RootTileInfo, 2> roots_;

  // Custom emitter-specific constraints to check in
  // `ParametersSatisfyConstraints`.
  std::vector<CustomConstraints> custom_constraints_;

  // Shape of the root instruction.
  Shape root_shape_;

  se::DeviceDescription device_info_;

  std::unique_ptr<TiledEmitterConstraints> tiled_emitter_constraints_;
};

namespace experimental {

// Verifies the Triton emitter constraints for a concrete tiling.
// `block_level_parameters` provides the launch parameters (num_stages /
// num_warps) used by the shared-memory estimate.
Decision VerifyTritonConstraints(
    const TiledHloComputation& tiled_computation,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters = {});

}  // namespace experimental

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TRITON_EMITTER_CONSTRAINTS_H_
