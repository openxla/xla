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

"""Canonical shape and benchmark configurations for KDA forward kernels."""

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

RCP_LN2 = 1.0 / math.log(2)


@dataclasses.dataclass(frozen=True)
class KdaBenchmarkConfig:
  """Configuration for a Kimi Delta Attention (KDA) forward workload."""

  name: str
  batch_size: int = 1
  seq_len: int = 8192
  heads: int = 32
  key_dim: int = 128
  value_dim: int = 128
  chunk_size: int = 64
  dtype: Any = jnp.bfloat16
  use_qk_l2norm: bool = True
  use_gate_in_kernel: bool = True
  safe_gate: bool = True
  lower_bound: float | None = -5.0
  use_initial_state: bool = False
  output_final_state: bool = False
  seq_lens: tuple[int, ...] | None = None
  batch_seq_lens: tuple[tuple[int, ...], ...] | None = None
  max_num_segments: int | None = None
  n_block_size: int = 8

  @property
  def is_varlen(self) -> bool:
    return self.seq_lens is not None or self.batch_seq_lens is not None

  @property
  def layouts(self) -> tuple[tuple[int, ...], ...] | None:
    if self.batch_seq_lens is not None:
      return self.batch_seq_lens
    if self.seq_lens is not None:
      return (self.seq_lens,)
    return None

  @property
  def effective_max_num_segments(self) -> int | None:
    if self.layouts is None:
      return None
    real_segs = max(len(s) for s in self.layouts)
    return max(real_segs, self.max_num_segments or 0)

  @property
  def num_chunks(self) -> int:
    return self.seq_len // self.chunk_size

  @property
  def scale(self) -> float:
    return float(self.key_dim**-0.5)

  @property
  def flops(self) -> float:
    """Returns estimated forward FLOPs for this KDA configuration."""
    bt = self.chunk_size
    k = self.key_dim
    v = self.value_dim
    nc = self.num_chunks
    intra_flops = (
        2 * bt * bt * k
        + 4 * bt * bt * k
        + 10 * 2 * bt * bt * bt
        + 2 * bt * bt * (v + k)
    )
    inter_flops = 4 * bt * k * v + 2 * bt * bt * v
    return float(
        self.batch_size * self.heads * nc * (intra_flops + inter_flops)
    )

  @property
  def bytes_moved(self) -> float:
    elem_bytes = 2 if self.dtype == jnp.bfloat16 else 4
    tokens = self.batch_size * self.heads * self.seq_len
    return float(
        tokens * (3 * self.key_dim + 2 * self.value_dim + 1) * elem_bytes
    )


KdaShapeConfig = KdaBenchmarkConfig


def make_balanced_segment_lengths(
    seq_len: int, num_segments: int
) -> tuple[int, ...]:
  """Builds balanced segment lengths summing to seq_len."""
  seg_len, rem = divmod(seq_len, num_segments)
  lengths = [seg_len + (1 if i < rem else 0) for i in range(num_segments)]
  return tuple(lengths)


KDA_BENCHMARK_CONFIGS: tuple[KdaBenchmarkConfig, ...] = (
    KdaBenchmarkConfig(
        name="kimi_fixed_b1_t2048_h32_d128_bf16",
        batch_size=1,
        seq_len=2048,
        heads=32,
        key_dim=128,
        value_dim=128,
        dtype=jnp.bfloat16,
        use_qk_l2norm=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    ),
    KdaBenchmarkConfig(
        name="kimi_fixed_b1_t8192_h32_d128_bf16",
        batch_size=1,
        seq_len=8192,
        heads=32,
        key_dim=128,
        value_dim=128,
        dtype=jnp.bfloat16,
        use_qk_l2norm=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    ),
    KdaBenchmarkConfig(
        name="kimi_packed_b1_t8192_n25_h32_d128_bf16",
        batch_size=1,
        seq_len=8192,
        heads=32,
        key_dim=128,
        value_dim=128,
        dtype=jnp.bfloat16,
        use_qk_l2norm=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        seq_lens=make_balanced_segment_lengths(8192, 25),
        max_num_segments=25,
    ),
    KdaBenchmarkConfig(
        name="kimi_fixed_b1_t1024_h8_d128_fp32",
        batch_size=1,
        seq_len=1024,
        heads=8,
        key_dim=128,
        value_dim=128,
        dtype=jnp.float32,
        use_qk_l2norm=False,
        use_gate_in_kernel=False,
        safe_gate=True,
        lower_bound=None,
        use_initial_state=True,
        output_final_state=True,
    ),
)


def make_segment_ids_from_layouts(
    layouts: tuple[tuple[int, ...], ...], seq_len: int
) -> jax.Array:
  """Builds 1 indexed segment IDs of shape [B, T] with 0 padding."""
  ids = np.zeros((len(layouts), seq_len), dtype=np.int32)
  for b_idx, seq_lens in enumerate(layouts):
    offset = 0
    for seg_id, length in enumerate(seq_lens, start=1):
      ids[b_idx, offset : offset + length] = seg_id
      offset += length
  return jnp.asarray(ids)


def _l2_normalize_init(x: jax.Array) -> jax.Array:
  x_f32 = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(jnp.square(x_f32), axis=-1) + 1e-6)
  return (x_f32 * rstd[..., None]).astype(x.dtype)


def make_kda_inputs(
    key: jax.Array,
    cfg: KdaBenchmarkConfig,
) -> dict[str, Any]:
  """Generates well conditioned KDA inputs for a KdaBenchmarkConfig."""
  keys = jax.random.split(key, 8)
  qk_shape = (cfg.heads, cfg.batch_size, cfg.seq_len, cfg.key_dim)
  v_shape = (cfg.heads, cfg.batch_size, cfg.seq_len, cfg.value_dim)
  beta_shape = (cfg.heads, cfg.batch_size, cfg.seq_len)

  segment_ids = (
      make_segment_ids_from_layouts(cfg.layouts, cfg.seq_len)
      if cfg.layouts is not None
      else None
  )
  valid = (
      segment_ids > 0
      if segment_ids is not None
      else jnp.ones((cfg.batch_size, cfg.seq_len), dtype=jnp.bool_)
  )
  valid_4d = valid[None, :, :, None]
  valid_3d = valid[None, :, :]

  q = jax.nn.silu(jax.random.normal(keys[0], qk_shape, dtype=jnp.float32))
  k = jax.nn.silu(jax.random.normal(keys[1], qk_shape, dtype=jnp.float32))
  if not cfg.use_qk_l2norm:
    q = _l2_normalize_init(q.astype(cfg.dtype))
    k = _l2_normalize_init(k.astype(cfg.dtype))

  v = 1.5 * jax.random.normal(keys[2], v_shape, dtype=jnp.float32)
  a_log = jax.random.uniform(
      keys[5], (cfg.heads,), minval=0.2, maxval=3.0, dtype=jnp.float32
  )
  delta_time_bias = jax.random.uniform(
      keys[6],
      (cfg.heads * cfg.key_dim,),
      minval=-8.0,
      maxval=-1.5,
      dtype=jnp.float32,
  )
  g_raw = jax.random.uniform(
      keys[3], qk_shape, minval=-4.5, maxval=4.5, dtype=jnp.float32
  )
  beta = jax.random.uniform(
      keys[4], beta_shape, minval=0.05, maxval=0.95, dtype=jnp.float32
  )

  if cfg.use_gate_in_kernel:
    g = g_raw
  else:
    gate_input = g_raw + delta_time_bias.reshape(cfg.heads, 1, 1, cfg.key_dim)
    a_val = jnp.exp(a_log).reshape(cfg.heads, 1, 1, 1)
    bound = cfg.lower_bound if cfg.lower_bound is not None else -5.0
    g = bound * jax.nn.sigmoid(a_val * gate_input)

  q = jnp.where(valid_4d, q, 0.0).astype(cfg.dtype)
  k = jnp.where(valid_4d, k, 0.0).astype(cfg.dtype)
  v = jnp.where(valid_4d, v, 0.0).astype(cfg.dtype)
  g = jnp.where(valid_4d, g, 0.0).astype(cfg.dtype)
  beta = jnp.where(valid_3d, beta, 0.0).astype(cfg.dtype)

  state_count = cfg.effective_max_num_segments or 1
  initial_state = None
  if cfg.use_initial_state:
    initial_state = 0.1 * jax.random.normal(
        keys[7],
        (cfg.batch_size, state_count, cfg.heads, cfg.key_dim, cfg.value_dim),
        dtype=jnp.float32,
    )

  return {
      "q": q,
      "k": k,
      "v": v,
      "g": g,
      "beta": beta,
      "a_log": a_log if cfg.use_gate_in_kernel else None,
      "delta_time_bias": delta_time_bias if cfg.use_gate_in_kernel else None,
      "scale": cfg.scale,
      "initial_state": initial_state,
      "output_final_state": cfg.output_final_state,
      "use_qk_l2norm": cfg.use_qk_l2norm,
      "use_gate_in_kernel": cfg.use_gate_in_kernel,
      "segment_ids": segment_ids,
      "safe_gate": cfg.safe_gate,
      "lower_bound": cfg.lower_bound,
      "max_num_segments": cfg.effective_max_num_segments,
      "chunk_size": cfg.chunk_size,
  }


_make_kda_inputs = make_kda_inputs
