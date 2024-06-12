/*
 Copyright 2024 CNAEIT

 Licensed under the Apache License, Version 2.0 (the "License");
 you y not use this file except in compliance with the License.
 You y obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef XLA_SERVICE_SPMD__SLICE_AUTO_SHARDED_STAGES_H_
#define XLA_SERVICE_SPMD__SLICE_AUTO_SHARDED_STAGES_H_

#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace spmd {

class SliceAutoShardedStages : public HloModulePass {
 public:
  SliceAutoShardedStages() = default;
  ~SliceAutoShardedStages() override = default;
  absl::string_view name() const override {
    return "slice_auto_sharded_stages";
  }

  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace spmd
}  // namespace xla


#endif  // XLA_SERVICE_SPMD__SLICE_AUTO_SHARDED_STAGES_H_