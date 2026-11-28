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

"""Canonical shape and benchmark configurations for Mamba2 SSD kernels."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Mamba2BenchmarkConfig:
  """Configuration for a Mamba2 SSD benchmark workload."""

  name: str
  batch_size: int
  seq_len: int
  d_inner: int
  n_heads: int
  d_state: int
  block_len: int
  n_block_size: int = 64

  @property
  def d_head(self) -> int:
    return self.d_inner // self.n_heads

  @property
  def num_chunks(self) -> int:
    return self.seq_len // self.block_len

  @property
  def flops(self) -> float:
    return float(
        self.batch_size
        * self.num_chunks
        * (
            2 * self.block_len * self.block_len * self.d_state
            + 2 * self.n_heads * self.block_len * self.block_len * self.d_head
            + 4 * self.n_heads * self.block_len * self.d_state * self.d_head
        )
    )

  @property
  def bytes_moved(self) -> float:
    return float(
        self.batch_size
        * self.seq_len
        * (2 * self.d_inner + 2 * self.d_state + self.n_heads)
        * 4
    )


Mamba2ShapeConfig = Mamba2BenchmarkConfig

MAMBA2_BENCHMARK_CONFIGS: tuple[Mamba2BenchmarkConfig, ...] = (
    Mamba2BenchmarkConfig(
        name="mamba2_4k_q64",
        batch_size=1,
        seq_len=4096,
        d_inner=512,
        n_heads=8,
        d_state=64,
        block_len=64,
        n_block_size=64,
    ),
    Mamba2BenchmarkConfig(
        name="mamba2_4k_q256",
        batch_size=1,
        seq_len=4096,
        d_inner=512,
        n_heads=8,
        d_state=64,
        block_len=256,
        n_block_size=64,
    ),
    Mamba2BenchmarkConfig(
        name="mamba2_16k_q256",
        batch_size=1,
        seq_len=16384,
        d_inner=512,
        n_heads=8,
        d_state=64,
        block_len=256,
        n_block_size=64,
    ),
)


def _make_ssd_inputs(
    key: jax.Array,
    *,
    batch_size: int,
    seq_len: int,
    d_inner: int,
    n_heads: int,
    d_state: int,
    dtype: Any = jnp.float32,
) -> dict[str, jax.Array]:
  """Generates well conditioned Mamba2 SSD inputs and parameters."""
  kx, kdt, kb, kc, ka, _, kbias = jax.random.split(key, 7)
  x = jax.random.normal(kx, (batch_size, seq_len, d_inner), dtype=dtype) * 0.2
  dt = jax.random.normal(kdt, (batch_size, seq_len, n_heads), dtype=dtype) * 0.1
  b = jax.random.normal(kb, (batch_size, seq_len, d_state), dtype=dtype) * 0.2
  c = jax.random.normal(kc, (batch_size, seq_len, d_state), dtype=dtype) * 0.2
  log_a = jnp.log(
      jax.random.uniform(
          ka, (n_heads,), minval=1.0, maxval=16.0, dtype=jnp.float32
      )
  ).astype(dtype)
  d = jnp.ones((n_heads,), dtype=dtype)
  dt_bias = jax.random.uniform(
      kbias, (n_heads,), minval=-4.0, maxval=-2.0, dtype=jnp.float32
  ).astype(dtype)
  return {
      "dt": dt,
      "x": x,
      "b": b,
      "c": c,
      "log_a": log_a,
      "d": d,
      "dt_bias": dt_bias,
  }


make_ssd_inputs = _make_ssd_inputs
