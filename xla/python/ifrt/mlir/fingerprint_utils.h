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

#ifndef XLA_PYTHON_IFRT_MLIR_FINGERPRINT_UTILS_H_
#define XLA_PYTHON_IFRT_MLIR_FINGERPRINT_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"

namespace xla {
namespace ifrt {

// Options for `FingerprintModuleOp`.
struct FingerprintOptions {
  // If true, the module name is canonicalized: it is fingerprinted as a fixed
  // ID, so modules that differ only in their names get the same fingerprint.
  bool canonicalize_module_name = false;

  // If true, the names of the Shardy meshes (`sdy.mesh`) defined in the module
  // body are canonicalized: each mesh name is fingerprinted as the rank of the
  // name among the mesh names, so mesh definitions, mesh uses and the order of
  // mesh names still contribute. A mesh keeps its name if a nested symbol table
  // defines a symbol with the same name.
  bool canonicalize_sdy_mesh_names = false;
};

// Returns a fingerprint of the given MLIR module. Two MLIR modules are
// equivalent if their fingerprints are the same, up to the names that
// `options` canonicalizes. May ignore debug info.
absl::StatusOr<uint64_t> FingerprintModuleOp(mlir::ModuleOp module_op,
                                             const FingerprintOptions& options);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_MLIR_FINGERPRINT_UTILS_H_
