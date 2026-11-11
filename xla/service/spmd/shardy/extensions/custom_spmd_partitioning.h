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

#ifndef XLA_SERVICE_SPMD_SHARDY_EXTENSIONS_CUSTOM_SPMD_PARTITIONING_H_
#define XLA_SERVICE_SPMD_SHARDY_EXTENSIONS_CUSTOM_SPMD_PARTITIONING_H_

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Names the custom-call target emitted by JAX `@custom_partitioning`.
inline constexpr llvm::StringLiteral kCustomSPMDPartitioningTargetName =
    "CustomSPMDPartitioning";

// Registers dialects and extensions required to deserialize, inline, and
// legalize MLIR modules returned by custom call partitioning callbacks.
void registerCustomSPMDPartitioningDialects(mlir::DialectRegistry& registry);

// Returns true if `moduleOp` contains any `stablehlo.custom_call` whose
// registered `xla::CustomCallPartitioner` supports MLIR partitioning
// (`CanPartitionToMlir()`).
bool hasCustomCallMlirPartitioner(mlir::ModuleOp moduleOp);

// Creates a pass that lowers `stablehlo.custom_call` ops with a registered
// MLIR `CustomCallPartitioner` into `sdy.manual_computation` (plus boundary
// reshards and divisibility pad/slice ops).
std::unique_ptr<mlir::Pass> createXlaSdyCustomCallPartitioningPass();

// Registers the `xla-sdy-custom-call-partitioning` pass.
void registerXlaSdyCustomCallPartitioningPass();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_EXTENSIONS_CUSTOM_SPMD_PARTITIONING_H_
