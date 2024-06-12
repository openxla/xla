/*
 Copyright 2024 CNAEIT

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#include "xla/service/spmd/slice_auto_sharded_stages.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_pipeline.h"

namespace xla {
namespace spmd {
Status RunAutoShardingPass(HloModule* hlo_module,
                           const CompileOptions& options) {
  TF_ASSIGN_OR_RETURN(auto module_config,
                      CreateHloModuleConfig(hlo_module, options));
  hlo_module->set_config(module_config);
  DumpHloModuleIfEnabled(*hlo_module, kBeforeAutoShardingDumpName);

  const DebugOptions& debug_options = hlo_module->config().debug_options();

  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts({},
                                                             ConvIsLowerable);
  layout_insensitive_algsimp_opts.set_minmax_propagate_nan(
      !debug_options.xla_gpu_enable_fast_min_max());
  layout_insensitive_algsimp_opts.set_enable_dot_strength_reduction(false);

  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("run-auto-sharding");
    AddHloVerifier(&spmd_pipeline);
    const int64_t num_partitions = hlo_module->config().num_partitions();
    spmd_pipeline.AddPass<CallInliner>();
    spmd_pipeline.AddPass<SliceAutoShardedStages>();
    // Remove redundant sharding ops when partition_count == 1.
    spmd_pipeline.AddPass<ShardingRemover>();
    spmd_pipeline.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }
  return OkStatus();
}  // namespace spmd
}  // namespace xla