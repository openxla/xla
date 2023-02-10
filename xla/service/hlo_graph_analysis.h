/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_ANALYSIS_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Returns a set of nodes that are considered activations. The inputs will not
// be considered as activations with the current implementation.
ConstHloInstructionSet ComputeHloActivationAnalysis(const HloModule* module);

// Defines a type for MatMul classes.
enum class MatmulClass {
  kWeightActivation,
  kActivationActivation,
  kActivationGradient,
  kWeightGradient,
  kUnimplemented
};

// Given a HLO module, returns a hash table including all matmauls and their
// respective classes.
absl::flat_hash_map<std::string, MatmulClass> ComputeHloMatmulClassifier(
    const HloModule* module);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_ANALYSIS_H_
