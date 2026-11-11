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

"""Plain JAX kda champion kernel package."""

from xla.jax_kernels.kda import champion
from xla.jax_kernels.kda import configs
from xla.jax_kernels.kda import reference

baseline = reference
KdaBenchmarkConfig = configs.KdaBenchmarkConfig
KdaShapeConfig = configs.KdaShapeConfig
KDA_BENCHMARK_CONFIGS = configs.KDA_BENCHMARK_CONFIGS
make_kda_inputs = configs.make_kda_inputs
make_segment_ids_from_layouts = configs.make_segment_ids_from_layouts
kda_reference = reference.kda_reference
kda_ref = reference.kda_ref
KdaResiduals = champion.KdaResiduals
kda_fwd_intra_grid = champion.kda_fwd_intra_grid
jax_k_kda_fwd_intra_fused = champion.jax_k_kda_fwd_intra_fused
kda_fwd_intra_fused = champion.kda_fwd_intra_fused
kda_fwd_pre_process_grid = champion.kda_fwd_pre_process_grid
chunk_gated_delta_rule_fwd_h_pre_process = (
    champion.chunk_gated_delta_rule_fwd_h_pre_process
)
kda_fwd_h_o_grid = champion.kda_fwd_h_o_grid
chunk_kda_fwd_h_o_varlen = champion.chunk_kda_fwd_h_o_varlen
kda_fwd_fused_grid = champion.kda_fwd_fused_grid
chunk_kda_fwd_custom = champion.chunk_kda_fwd_custom
kda_fwd_jax_k = champion.kda_fwd_jax_k
RCP_LN2 = reference.RCP_LN2
align_seqs = reference.align_seqs
align_up = reference.align_up
cdiv = reference.cdiv
exp2 = reference.exp2
l2norm_fwd = reference.l2norm_fwd
prepare_chunk_indices = reference.prepare_chunk_indices
reference_chunk_gated_delta_rule_fwd_h_pre_process = (
    reference.reference_chunk_gated_delta_rule_fwd_h_pre_process
)
reference_chunk_kda_fwd_h_o_varlen = (
    reference.reference_chunk_kda_fwd_h_o_varlen
)
reference_kda_fwd_intra_fused = reference.reference_kda_fwd_intra_fused
segment_ids_to_cu_seqlens = reference.segment_ids_to_cu_seqlens
segment_ids_to_seqlens = reference.segment_ids_to_seqlens
unalign_output = reference.unalign_output

__all__ = [
    "KDA_BENCHMARK_CONFIGS",
    "KdaBenchmarkConfig",
    "KdaResiduals",
    "KdaShapeConfig",
    "RCP_LN2",
    "align_seqs",
    "align_up",
    "baseline",
    "cdiv",
    "champion",
    "chunk_gated_delta_rule_fwd_h_pre_process",
    "chunk_kda_fwd_custom",
    "chunk_kda_fwd_h_o_varlen",
    "configs",
    "exp2",
    "jax_k_kda_fwd_intra_fused",
    "kda_fwd_fused_grid",
    "kda_fwd_h_o_grid",
    "kda_fwd_intra_fused",
    "kda_fwd_intra_grid",
    "kda_fwd_jax_k",
    "kda_fwd_pre_process_grid",
    "kda_ref",
    "kda_reference",
    "l2norm_fwd",
    "make_kda_inputs",
    "make_segment_ids_from_layouts",
    "prepare_chunk_indices",
    "reference",
    "reference_chunk_gated_delta_rule_fwd_h_pre_process",
    "reference_chunk_kda_fwd_h_o_varlen",
    "reference_kda_fwd_intra_fused",
    "segment_ids_to_cu_seqlens",
    "segment_ids_to_seqlens",
    "unalign_output",
]
