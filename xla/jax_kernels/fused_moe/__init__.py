# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plain JAX fused_moe champion kernel package."""

from xla.jax_kernels.fused_moe import champion
from xla.jax_kernels.fused_moe import configs
from xla.jax_kernels.fused_moe import reference

baseline = reference
FUSED_MOE_BENCHMARK_CONFIGS = configs.FUSED_MOE_BENCHMARK_CONFIGS
FusedMoeBenchmarkConfig = configs.FusedMoeBenchmarkConfig
FusedMoeTilingConfig = configs.FusedMoeTilingConfig
TUNED_BLOCK_SIZES = configs.TUNED_BLOCK_SIZES
align_to = configs.align_to
cdiv = configs.cdiv
create_default_mesh = configs.create_default_mesh
gen_moe_inputs = configs.gen_moe_inputs
get_default_block_sizes = configs.get_default_block_sizes
get_dtype_packing = configs.get_dtype_packing
get_tuned_block_sizes = configs.get_tuned_block_sizes
make_moe_inputs = configs.make_moe_inputs
sub_channel_quantize = configs.sub_channel_quantize
moe_reference = reference.ref_moe
apply_act_fn = champion.apply_act_fn
apply_scoring_fn = champion.apply_scoring_fn
fused_ep_moe = champion.fused_ep_moe
fused_moe_jax_k = champion.fused_ep_moe
fused_moe_tile_pipeline = champion.fused_moe_tile_pipeline
jax_k_fused_ep_moe = champion.fused_ep_moe
ref_moe = champion.ref_moe
swigluoai = champion.swigluoai

__all__ = [
    "FUSED_MOE_BENCHMARK_CONFIGS",
    "FusedMoeBenchmarkConfig",
    "FusedMoeTilingConfig",
    "TUNED_BLOCK_SIZES",
    "align_to",
    "apply_act_fn",
    "apply_scoring_fn",
    "baseline",
    "cdiv",
    "champion",
    "configs",
    "create_default_mesh",
    "fused_ep_moe",
    "fused_moe_jax_k",
    "fused_moe_tile_pipeline",
    "gen_moe_inputs",
    "get_default_block_sizes",
    "get_dtype_packing",
    "get_tuned_block_sizes",
    "jax_k_fused_ep_moe",
    "make_moe_inputs",
    "moe_reference",
    "ref_moe",
    "reference",
    "sub_channel_quantize",
    "swigluoai",
]
