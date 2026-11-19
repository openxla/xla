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

"""Shape, block size, and benchmark configurations for jax_k fused MoE kernels."""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def cdiv(a: int, b: int) -> int:
  assert b != 0
  return (a + b - 1) // b


def align_to(x: int, a: int) -> int:
  return cdiv(x, a) * a


def itemsize_bits(dtype: Any) -> int:
  dt = jnp.dtype(dtype)
  if "float4" in dt.name or "int4" in dt.name:
    return 4
  if "int2" in dt.name:
    return 2
  return int(dt.itemsize * 8)


def get_dtype_packing(dtype: Any) -> int:
  bits = itemsize_bits(dtype)
  return 32 // bits


TUNED_BLOCK_SIZES: dict[
    tuple[int, int, int, int, int, int, int, int],
    tuple[int, int, int, int, int, int, int, int],
] = {
    (3072, 3072, 128, 4, 2, 2, 128, 4): (
        32,
        256 * 6,
        256 * 12,
        256 * 12,
        16,
        256 * 6,
        256 * 12,
        256 * 12,
    ),
    (3072, 3072, 128, 4, 2, 2, 128, 8): (
        16,
        256 * 6,
        256 * 6,
        256 * 12,
        16,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 256, 4): (
        64,
        256 * 6,
        256 * 6,
        256 * 12,
        16,
        256 * 6,
        256 * 6,
        256 * 12,
    ),
    (3072, 3072, 128, 4, 2, 2, 256, 8): (
        32,
        256 * 6,
        256 * 6,
        256 * 12,
        32,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 512, 4): (
        128,
        256 * 6,
        256 * 6,
        256 * 6,
        64,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 512, 8): (
        64,
        256 * 6,
        256 * 6,
        256 * 12,
        64,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 1024, 4): (
        128,
        256 * 3,
        256 * 6,
        256 * 6,
        64,
        256 * 3,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 1024, 8): (
        64,
        256 * 6,
        256 * 6,
        256 * 12,
        64,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 2048, 4): (
        128,
        256 * 3,
        256 * 6,
        256 * 6,
        64,
        256 * 3,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 2048, 8): (
        64,
        256 * 6,
        256 * 6,
        256 * 12,
        64,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (6144, 2560, 160, 8, 2, 4, 16, 4): (
        256 * 1,
        256 * 5,
        256 * 12,
        256 * 24,
        16,
        256 * 5,
        256 * 12,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 16, 8): (
        16,
        256 * 10,
        256 * 8,
        256 * 12,
        16,
        256 * 2,
        256 * 8,
        256 * 4,
    ),
    (6144, 2560, 160, 8, 2, 4, 32, 4): (
        128,
        256 * 5,
        256 * 12,
        256 * 24,
        32,
        256 * 5,
        256 * 12,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 32, 8): (
        64,
        256 * 10,
        256 * 4,
        256 * 6,
        32,
        256 * 10,
        256 * 4,
        256 * 2,
    ),
    (6144, 2560, 160, 8, 2, 4, 64, 4): (
        64,
        256 * 10,
        256 * 12,
        256 * 12,
        32,
        256 * 10,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 64, 8): (
        32,
        256 * 10,
        256 * 6,
        256 * 12,
        32,
        256 * 5,
        256 * 6,
        256 * 3,
    ),
    (6144, 2560, 160, 8, 2, 4, 128, 4): (
        64,
        256 * 5,
        256 * 24,
        256 * 24,
        16,
        256 * 5,
        256 * 24,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 128, 8): (
        16,
        256 * 5,
        256 * 12,
        256 * 24,
        16,
        256 * 1,
        256 * 12,
        256 * 8,
    ),
    (6144, 2560, 160, 8, 2, 4, 256, 4): (
        256 * 1,
        256 * 10,
        256 * 8,
        256 * 8,
        32,
        256 * 10,
        256 * 8,
        256 * 8,
    ),
    (6144, 2560, 160, 8, 2, 4, 256, 8): (
        32,
        256 * 10,
        256 * 8,
        256 * 12,
        32,
        256 * 10,
        256 * 8,
        256 * 4,
    ),
    (6144, 2560, 160, 8, 2, 4, 512, 4): (
        256 * 1,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 512, 8): (
        64,
        256 * 2,
        256 * 24,
        256 * 24,
        32,
        256 * 1,
        256 * 24,
        256 * 8,
    ),
    (6144, 2560, 160, 8, 2, 4, 1024, 4): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 1024, 8): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 4,
    ),
    (6144, 2560, 160, 8, 2, 4, 2048, 4): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 2048, 8): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 4096, 4): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 4096, 8): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 6,
    ),
    (6144, 2560, 160, 8, 2, 4, 8192, 4): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        32,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 8192, 8): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 6,
    ),
}


def _largest_aligned_divisor(dim: int, cap: int, align: int = 128) -> int:
  step = max(1, min(dim, align))
  for cand in range(min(dim, cap), step - 1, -step):
    if dim % cand == 0 and cand % step == 0:
      return cand
  return min(dim, step)


def get_default_block_sizes(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    t_packing: int,
    w_packing: int,
    num_tokens: int,
    ep_size: int,
) -> tuple[int, int, int, int, int, int, int, int]:
  """Returns default block sizes for fused MoE when not in TUNED_BLOCK_SIZES."""
  assert num_experts % ep_size == 0
  assert num_tokens % ep_size == 0
  assert 0 < top_k <= num_experts
  assert 0 < t_packing <= 8
  assert 0 < w_packing <= 8
  if hidden_size % 256:
    hidden_size = align_to(hidden_size, 256)
  if intermediate_size % 256:
    intermediate_size = align_to(intermediate_size, 256)
  local_num_tokens = num_tokens // ep_size

  d = hidden_size // 256
  f = intermediate_size // 256
  bt = _largest_aligned_divisor(
      local_num_tokens, min(local_num_tokens, 128), max(t_packing, 8)
  )
  btc = _largest_aligned_divisor(bt, min(max(bt // 2, 8), 64), min(bt, 8))
  bf = _largest_aligned_divisor(intermediate_size, min(256 * f // 2, 1024), 128)
  bd1 = _largest_aligned_divisor(
      hidden_size, min(256 * d // 2, 1024), t_packing * 128
  )
  bd2 = _largest_aligned_divisor(
      hidden_size, min(256 * d // 2, 2048), t_packing * 128
  )

  return (
      bt,
      bf,
      bd1,
      bd2,
      btc,
      bf,
      bd1,
      bd2,
  )


def get_tuned_block_sizes(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    t_packing: int,
    w_packing: int,
    num_tokens: int,
    ep_size: int,
) -> tuple[int, int, int, int, int, int, int, int]:
  """Looks up tuned block sizes or falls back to default block sizes."""
  assert num_experts % ep_size == 0
  assert num_tokens % ep_size == 0
  assert 0 < top_k <= num_experts
  assert 0 < t_packing <= 8
  assert 0 < w_packing <= 8
  hidden_size = align_to(hidden_size, 256)
  intermediate_size = align_to(intermediate_size, 256)

  key = (
      hidden_size,
      intermediate_size,
      num_experts,
      top_k,
      t_packing,
      w_packing,
      num_tokens,
      ep_size,
  )
  if key in TUNED_BLOCK_SIZES:
    return TUNED_BLOCK_SIZES[key]
  for fallback_ep in (8, 4):
    alt_key = (
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        t_packing,
        w_packing,
        num_tokens,
        fallback_ep,
    )
    if alt_key in TUNED_BLOCK_SIZES:
      bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c = TUNED_BLOCK_SIZES[alt_key]
      local_num_tokens = num_tokens // ep_size
      bt = _largest_aligned_divisor(
          local_num_tokens, min(local_num_tokens, bt), max(t_packing, 8)
      )
      btc = _largest_aligned_divisor(bt, min(bt, btc), min(bt, 8))
      return (bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c)
  return get_default_block_sizes(*key)


@dataclasses.dataclass(frozen=True)
class FusedMoeTilingConfig:
  """Stage 1 jax_k tiling configuration for fused MoE."""

  tile_m: int
  bf: int
  vmem_kib: int

  @property
  def is_valid(self) -> bool:
    return self.tile_m >= 8 and self.bf >= 128

  @classmethod
  def from_shapes(
      cls,
      num_tokens: int,
      hidden_size: int,
      intermediate_size: int,
      num_experts: int,
      top_k: int,
      w_dtype: Any,
      subc_quant_w2_sz: int | None = None,
      is_tpu7x: bool = True,
      scoring_fn: str = "softmax",
  ) -> "FusedMoeTilingConfig":
    """Selects VMEM aware (tile_m, bf, vmem_kib) tiles for jax_k fused MoE."""
    avg_tokens = max(1, (num_tokens * top_k) // max(1, num_experts))
    if scoring_fn == "sigmoid":
      tile_m = min(align_to(num_tokens, 16), 128)
    elif hidden_size >= 3072 and intermediate_size >= 2560:
      if avg_tokens <= 8:
        tile_m = 16
      elif avg_tokens <= 20:
        tile_m = 32
      else:
        tile_m = 64
    else:
      if avg_tokens <= 8:
        tile_m = 16
      elif avg_tokens <= 16:
        tile_m = 32
      elif avg_tokens <= 32:
        tile_m = 64
      else:
        tile_m = 128

    w_bits = itemsize_bits(w_dtype)
    max_weight_bytes = 56 * 1024 * 1024
    min_bf_align = 128
    if subc_quant_w2_sz is not None and subc_quant_w2_sz < intermediate_size:
      min_bf_align = subc_quant_w2_sz

    chosen_bf = min_bf_align
    for cand_bf in (
        intermediate_size,
        1536,
        1280,
        1024,
        768,
        640,
        512,
        384,
        256,
        128,
    ):
      if (
          cand_bf <= intermediate_size
          and intermediate_size % cand_bf == 0
          and cand_bf % min_bf_align == 0
      ):
        step_weight_bytes = (3 * hidden_size * cand_bf * w_bits) // 8
        if step_weight_bytes <= max_weight_bytes:
          chosen_bf = cand_bf
          break

    del is_tpu7x
    vmem_kib = 32768
    return cls(tile_m=tile_m, bf=chosen_bf, vmem_kib=vmem_kib)


@dataclasses.dataclass(frozen=True)
class FusedMoeBenchmarkConfig:
  """Configuration for a fused MoE benchmark workload."""

  name: str
  num_tokens: int
  hidden_size: int
  intermediate_size: int
  num_experts: int
  top_k: int
  dtype: Any = jnp.bfloat16
  w_dtype: Any = None
  renormalize_topk_logits: bool = True
  act_fn: str = "silu"
  scoring_fn: str = "softmax"
  subc_quant_w1_sz: int | None = None
  subc_quant_w2_sz: int | None = None
  has_bias: bool = False
  bt: int | None = None
  bf: int | None = None
  bd1: int | None = None
  bd2: int | None = None
  btc: int | None = None
  bfc: int | None = None
  bd1c: int | None = None
  bd2c: int | None = None
  seed: int = 1234

  @property
  def tokens_shape(self) -> tuple[int, int]:
    return (self.num_tokens, self.hidden_size)

  @property
  def w1_shape(self) -> tuple[int, int, int, int]:
    return (self.num_experts, 2, self.hidden_size, self.intermediate_size)

  @property
  def w2_shape(self) -> tuple[int, int, int]:
    return (self.num_experts, self.intermediate_size, self.hidden_size)

  @property
  def gating_shape(self) -> tuple[int, int]:
    return (self.num_tokens, self.num_experts)

  @property
  def flops(self) -> float:
    return float(
        6
        * self.num_tokens
        * self.top_k
        * self.hidden_size
        * self.intermediate_size
    )


FUSED_MOE_BENCHMARK_CONFIGS: tuple[FusedMoeBenchmarkConfig, ...] = (
    FusedMoeBenchmarkConfig(
        name="basic_t256_h1024_f1024_e128_k8_bf16",
        num_tokens=256,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=128,
        top_k=8,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="softmax",
        bt=32,
        bf=1024,
        bd1=1024,
        bd2=1024,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="qwen3_30b_a3b_t512_h2048_f768_e128_k8_bf16",
        num_tokens=512,
        hidden_size=2048,
        intermediate_size=768,
        num_experts=128,
        top_k=8,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="softmax",
        bt=16,
        bf=384,
        bd1=512,
        bd2=512,
        btc=16,
        bfc=384,
        bd1c=256,
        bd2c=256,
        seed=54321,
    ),
    FusedMoeBenchmarkConfig(
        name="qwen3_235b_t512_h4096_f1536_e128_k8_bf16",
        num_tokens=512,
        hidden_size=4096,
        intermediate_size=1536,
        num_experts=128,
        top_k=8,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="softmax",
        bt=64,
        bf=768,
        bd1=2048,
        bd2=2048,
        btc=64,
        bfc=768,
        bd1c=2048,
        bd2c=2048,
        seed=54321,
    ),
    FusedMoeBenchmarkConfig(
        name="gpt_oss_t128_h3072_f3072_e128_k4_bf16",
        num_tokens=128,
        hidden_size=3072,
        intermediate_size=3072,
        num_experts=128,
        top_k=4,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="softmax",
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="gpt_oss_t256_h3072_f3072_e128_k4_bf16",
        num_tokens=256,
        hidden_size=3072,
        intermediate_size=3072,
        num_experts=128,
        top_k=4,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="softmax",
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="gpt_oss_t512_h3072_f3072_e128_k4_bf16",
        num_tokens=512,
        hidden_size=3072,
        intermediate_size=3072,
        num_experts=128,
        top_k=4,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="softmax",
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="qwen3_coder_t128_h6144_f2560_e160_k8_fp8",
        num_tokens=128,
        hidden_size=6144,
        intermediate_size=2560,
        num_experts=160,
        top_k=8,
        w_dtype=jnp.float8_e5m2,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="softmax",
        subc_quant_w1_sz=6144,
        subc_quant_w2_sz=2560,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="qwen3_coder_t256_h6144_f2560_e160_k8_fp8",
        num_tokens=256,
        hidden_size=6144,
        intermediate_size=2560,
        num_experts=160,
        top_k=8,
        w_dtype=jnp.float8_e5m2,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="softmax",
        subc_quant_w1_sz=6144,
        subc_quant_w2_sz=2560,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="qwen3_coder_t512_h6144_f2560_e160_k8_fp8",
        num_tokens=512,
        hidden_size=6144,
        intermediate_size=2560,
        num_experts=160,
        top_k=8,
        w_dtype=jnp.float8_e5m2,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="softmax",
        subc_quant_w1_sz=6144,
        subc_quant_w2_sz=2560,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="subc_quant_int8_t256_h1024_f1024_e128_k8",
        num_tokens=256,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=128,
        top_k=8,
        w_dtype=jnp.int8,
        renormalize_topk_logits=False,
        act_fn="silu",
        scoring_fn="softmax",
        subc_quant_w1_sz=256,
        subc_quant_w2_sz=256,
        bt=32,
        bf=1024,
        bd1=1024,
        bd2=1024,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="subc_quant_fp8_t256_h1024_f1024_e128_k8",
        num_tokens=256,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=128,
        top_k=8,
        w_dtype=jnp.float8_e5m2,
        renormalize_topk_logits=False,
        act_fn="silu",
        scoring_fn="softmax",
        subc_quant_w1_sz=256,
        subc_quant_w2_sz=256,
        bt=32,
        bf=1024,
        bd1=1024,
        bd2=1024,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="subc_quant_fp4_t256_h1024_f1024_e128_k8",
        num_tokens=256,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=128,
        top_k=8,
        w_dtype=jnp.float4_e2m1fn,
        renormalize_topk_logits=False,
        act_fn="silu",
        scoring_fn="softmax",
        subc_quant_w1_sz=256,
        subc_quant_w2_sz=256,
        bt=32,
        bf=1024,
        bd1=1024,
        bd2=1024,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="perc_quant_int8_t256_h512_f1024_e128_k8",
        num_tokens=256,
        hidden_size=512,
        intermediate_size=1024,
        num_experts=128,
        top_k=8,
        w_dtype=jnp.int8,
        renormalize_topk_logits=False,
        act_fn="silu",
        scoring_fn="softmax",
        subc_quant_w1_sz=512,
        subc_quant_w2_sz=1024,
        bt=32,
        bf=1024,
        bd1=512,
        bd2=512,
        btc=32,
        bfc=512,
        bd1c=256,
        bd2c=256,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="perc_quant_fp8_t256_h512_f1024_e128_k8",
        num_tokens=256,
        hidden_size=512,
        intermediate_size=1024,
        num_experts=128,
        top_k=8,
        w_dtype=jnp.float8_e5m2,
        renormalize_topk_logits=False,
        act_fn="silu",
        scoring_fn="softmax",
        subc_quant_w1_sz=512,
        subc_quant_w2_sz=1024,
        bt=32,
        bf=1024,
        bd1=512,
        bd2=512,
        btc=32,
        bfc=512,
        bd1c=256,
        bd2c=256,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="bias_swigluoai_t256_h1024_f1024_e128_k8",
        num_tokens=256,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=128,
        top_k=8,
        renormalize_topk_logits=True,
        act_fn="swigluoai",
        scoring_fn="softmax",
        has_bias=True,
        bt=32,
        bf=512,
        bd1=512,
        bd2=512,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="act_gelu_t256_h1024_f1024_e128_k8_bf16",
        num_tokens=256,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=128,
        top_k=8,
        renormalize_topk_logits=True,
        act_fn="gelu",
        scoring_fn="softmax",
        bt=32,
        bf=512,
        bd1=512,
        bd2=512,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
        seed=1234,
    ),
    FusedMoeBenchmarkConfig(
        name="scoring_sigmoid_t256_h1024_f1024_e128_k8_bf16",
        num_tokens=256,
        hidden_size=1024,
        intermediate_size=1024,
        num_experts=128,
        top_k=8,
        renormalize_topk_logits=True,
        act_fn="silu",
        scoring_fn="sigmoid",
        bt=32,
        bf=512,
        bd1=512,
        bd2=512,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
        seed=1234,
    ),
)


def create_default_mesh(ep_axis_name: str = "model") -> jax.sharding.Mesh:
  """Creates the canonical 2D ('data', ep_axis_name) TPU mesh."""
  mesh_devices = sorted(
      jax.devices(),
      key=lambda x: (
          x.coords[0],
          (-1 if x.coords[0] % 2 else 1) * x.coords[1],
      ),
  )
  return jax.sharding.Mesh(
      np.array(mesh_devices).reshape(1, -1),
      axis_names=("data", ep_axis_name),
  )


def gen_moe_inputs(
    dtype: Any,
    top_k: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_tokens: int,
    *,
    seed: int = 1234,
    has_bias: bool = False,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array,
]:
  """Generates deterministic MoE inputs with unique top k experts per token."""
  key = jax.random.key(seed)
  k0, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)

  a = (
      jax.random.normal(
          k0, (num_tokens, hidden_size), dtype=jnp.bfloat16
      ).astype(dtype)
      / 10
  )

  w1 = (
      jax.random.normal(
          k1,
          (num_experts, 2, hidden_size, intermediate_size),
          dtype=jnp.bfloat16,
      )
      / 10
  ).astype(dtype)
  w2 = (
      jax.random.normal(
          k2,
          (num_experts, intermediate_size, hidden_size),
          dtype=jnp.bfloat16,
      )
      / 10
  ).astype(dtype)

  if has_bias:
    b1 = (
        jax.random.normal(
            k3, (num_experts, 2, 1, intermediate_size), dtype=jnp.bfloat16
        )
        / 10
    ).astype(dtype)
    b2 = (
        jax.random.normal(k4, (num_experts, 1, hidden_size), dtype=jnp.bfloat16)
        / 10
    ).astype(dtype)
  else:
    b1 = b2 = None

  gating_output = (
      jax.random.normal(k5, (num_tokens, num_experts), dtype=jnp.bfloat16)
      + jnp.arange(num_tokens * num_experts, dtype=jnp.bfloat16).reshape(
          num_tokens, num_experts
      )
      / 100
  )

  top_k_indices = jax.random.randint(
      k6,
      (num_tokens, top_k),
      minval=0,
      maxval=num_experts - 1,
      dtype=jnp.int32,
  )

  one_hot = (
      jnp.sum(
          jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.bfloat16),
          axis=1,
      )
      * 30
  )

  gating_output = (gating_output + one_hot).astype(dtype)

  return a, w1, w2, b1, b2, gating_output


@functools.partial(jax.jit, static_argnames=("quant_dtype", "wsz"))
def sub_channel_quantize(
    x: jax.Array, quant_dtype: Any, wsz: int = 256
) -> tuple[jax.Array, jax.Array]:
  """Quantizes x with sub channel quantization along the second minor axis."""
  if jnp.issubdtype(quant_dtype, jnp.floating):
    dtype_info = jnp.finfo(quant_dtype)
  else:
    dtype_info = jnp.iinfo(quant_dtype)
  dtype_max = float(dtype_info.max)
  assert len(x.shape) >= 2
  assert x.shape[-2] % wsz == 0
  num_blocks = x.shape[-2] // wsz
  blocked = x.reshape(*x.shape[:-2], num_blocks, wsz, x.shape[-1])
  abs_max = jnp.abs(blocked).max(axis=-2, keepdims=True)
  scale = (abs_max / dtype_max).astype(jnp.float32)
  w = (blocked / scale).astype(quant_dtype).reshape(x.shape)
  return w, scale


@functools.partial(
    jax.jit, static_argnames=("quant_dtype", "wsz"), donate_argnums=(0,)
)
def _quantize_donate(
    x: jax.Array, quant_dtype: Any, wsz: int = 256
) -> tuple[jax.Array, jax.Array]:
  return sub_channel_quantize(x, quant_dtype, wsz)


def make_moe_inputs(
    cfg: FusedMoeBenchmarkConfig,
    mesh: jax.sharding.Mesh | None = None,
    ep_axis_name: str = "model",
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
  """Builds (tokens, w1, w2, gating_output, w1_scale, w2_scale, b1, b2) for cfg."""
  ep_sharding = (
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(ep_axis_name))
      if (mesh is not None and mesh.shape[ep_axis_name] > 1)
      else None
  )

  def _put(x: jax.Array) -> jax.Array:
    if ep_sharding is None:
      return x
    return jax.device_put(x, ep_sharding)

  key = jax.random.key(cfg.seed)
  k0, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)

  w1 = _put(
      (
          jax.random.normal(
              k1,
              (cfg.num_experts, 2, cfg.hidden_size, cfg.intermediate_size),
              dtype=jnp.bfloat16,
          )
          / 10
      ).astype(cfg.dtype)
  )
  w1_scale = None
  if cfg.w_dtype is not None:
    subc_w1 = cfg.subc_quant_w1_sz or 256
    w1, w1_scale = _quantize_donate(w1, cfg.w_dtype, subc_w1)
    jax.block_until_ready((w1, w1_scale))

  w2 = _put(
      (
          jax.random.normal(
              k2,
              (cfg.num_experts, cfg.intermediate_size, cfg.hidden_size),
              dtype=jnp.bfloat16,
          )
          / 10
      ).astype(cfg.dtype)
  )
  w2_scale = None
  if cfg.w_dtype is not None:
    subc_w2 = cfg.subc_quant_w2_sz or 256
    w2, w2_scale = _quantize_donate(w2, cfg.w_dtype, subc_w2)
    jax.block_until_ready((w2, w2_scale))

  tokens = _put((
      jax.random.normal(
          k0, (cfg.num_tokens, cfg.hidden_size), dtype=jnp.bfloat16
      ).astype(cfg.dtype)
      / 10
  ))
  if cfg.has_bias:
    b1 = _put(
        (
            jax.random.normal(
                k3,
                (cfg.num_experts, 2, 1, cfg.intermediate_size),
                dtype=jnp.bfloat16,
            )
            / 10
        ).astype(jnp.float32)
    )
    b2 = _put(
        (
            jax.random.normal(
                k4, (cfg.num_experts, 1, cfg.hidden_size), dtype=jnp.bfloat16
            )
            / 10
        ).astype(jnp.float32)
    )
  else:
    b1 = b2 = None

  gating_output = (
      jax.random.normal(
          k5, (cfg.num_tokens, cfg.num_experts), dtype=jnp.bfloat16
      )
      + jnp.arange(
          cfg.num_tokens * cfg.num_experts, dtype=jnp.bfloat16
      ).reshape(cfg.num_tokens, cfg.num_experts)
      / 100
  )
  top_k_indices = jax.random.randint(
      k6,
      (cfg.num_tokens, cfg.top_k),
      minval=0,
      maxval=cfg.num_experts - 1,
      dtype=jnp.int32,
  )
  one_hot = (
      jnp.sum(
          jax.nn.one_hot(top_k_indices, cfg.num_experts, dtype=jnp.bfloat16),
          axis=1,
      )
      * 30
  )
  gating_output = _put((gating_output + one_hot).astype(cfg.dtype))
  jax.block_until_ready((tokens, w1, w2, gating_output))
  return tokens, w1, w2, gating_output, w1_scale, w2_scale, b1, b2
