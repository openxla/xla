/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_ALL_GATHER_COMBINER_H_
#define XLA_SERVICE_ALL_GATHER_COMBINER_H_

#include <cstdint>
#include <string_view>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Combines small non-dependent AllGather ops into larger combined
// AllGather ops. A typical AllGather implementation has a minimum
// latency-induced time for a AllGather op so a single combined op can be
// more efficient than many small ones.
class AllGatherCombiner : public HloModulePass {
 public:
  AllGatherCombiner(int64_t combine_threshold_in_bytes,
                    int64_t combine_threshold_count, bool combine_by_dim,
                    bool is_async = false,
                    std::string_view async_strategy = "trivial",
                    int64_t near_op_threshold = 5);

  absl::string_view name() const override { return "all-gather-combiner"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Combine all gather ops up to this threshold.
  int64_t combine_threshold_in_bytes_;

  // Combine all gather ops up to this threshold (number of operands).
  int64_t combine_threshold_count_;

  // Combine only all-gather ops with the same gather dimension.
  bool combine_by_dim_;

  // Whether to operate on Async pairs. Only works post-scheduling.
  bool is_async_;

  AsyncCombinerStrategy async_strategy_;

  int64_t near_op_threshold_;
};

class AsyncAllGatherCombiner : public AllGatherCombiner {
 public:
  AsyncAllGatherCombiner(std::string_view strategy = "trivial",
                         int64_t near_op_threshold = 5)
      : AllGatherCombiner(
            /*combine_threshold_in_bytes=*/INT64_MAX,
            /*combine_threshold_count=*/INT64_MAX,
            /*combine_by_dim=*/false,
            /*is_async=*/true, strategy, near_op_threshold) {}

  absl::string_view name() const override {
    return "async-all-gather-combiner";
  }
};

}  // namespace xla

#endif  // XLA_SERVICE_ALL_GATHER_COMBINER_H_
