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

"""Configuration definitions for jax_k Splash Attention backward kernel."""

import dataclasses
import math

DEFAULT_MASK_VALUE = -1e30
LOG2E = math.log2(math.e)
LOG2E_INV = 1.0 / LOG2E
LN2 = math.log(2)


@dataclasses.dataclass(frozen=True)
class SplashBwdConfig:
  """Specification of a Splash Attention backward benchmark shape and tuning parameters.

  Attributes:
    case_name: Human readable label for the benchmark case.
    batch_size: Batch size dimension.
    q_seq_len: Query sequence length.
    kv_seq_len: Key and value sequence length.
    num_q_heads: Number of query attention heads.
    num_kv_heads: Number of key/value attention heads.
    dim_qk: Feature dimension for queries and keys.
    dim_v: Feature dimension for values.
    block_q: Query sequence tile size for jax_k kernel.
    block_kv: Key/value sequence tile size for jax_k kernel.
    block_q_head: Query head block size for GQA in jax_k kernel.
    block_kv_head: Key/value head block size for GQA in jax_k kernel.
    pl_block_q: Query tile size for Tokamax Pallas baseline.
    pl_block_kv: Key/value tile size for Tokamax Pallas baseline.
    pl_block_kv_compute: Compute tile size for Tokamax Pallas baseline.
    pl_block_q_dkv: Query DKV tile size for Tokamax Pallas baseline.
    pl_block_kv_dkv: Key/value DKV tile size for Tokamax Pallas baseline.
    pl_block_kv_dkv_compute: DKV compute tile size for Tokamax Pallas baseline.
    causal: Whether lower triangular causal masking applies.
    use_log2: Whether base 2 exponentiation is used.
    mask_value: Additive value applied to masked attention positions.
    attn_logits_soft_cap: Optional threshold for tanh logit soft capping.
  """

  case_name: str
  batch_size: int
  q_seq_len: int
  kv_seq_len: int
  num_q_heads: int
  num_kv_heads: int
  dim_qk: int
  dim_v: int
  block_q: int
  block_kv: int
  block_q_head: int
  block_kv_head: int
  pl_block_q: int
  pl_block_kv: int
  pl_block_kv_compute: int
  pl_block_q_dkv: int
  pl_block_kv_dkv: int
  pl_block_kv_dkv_compute: int
  causal: bool = True
  use_log2: bool = True
  mask_value: float = DEFAULT_MASK_VALUE
  attn_logits_soft_cap: float | None = None

  @property
  def is_gqa(self) -> bool:
    """Returns True when query head count differs from key/value head count."""
    return self.num_q_heads != self.num_kv_heads

  @property
  def is_mqa(self) -> bool:
    """Returns True when key/value head count is 1."""
    return self.num_kv_heads == 1

  @property
  def q_groups(self) -> int:
    """Returns the number of query heads per key/value head."""
    return self.num_q_heads // self.num_kv_heads

  @property
  def is_long_seq(self) -> bool:
    """Returns True when query sequence length exceeds 8192."""
    return self.q_seq_len > 8192

  @property
  def backward_flops(self) -> float:
    """Returns active attention backward FLOPs for this configuration.

    In backward attention, five matrix contractions are performed per active
    position: S = Q @ K.T (recomputed logits), dV = P.T @ dO, dP = dO @ V.T,
    dK = dS.T @ Q, and dQ = dS @ K.
    """
    active_fraction = 0.5 if self.causal else 1.0
    active_tokens = (
        self.batch_size
        * self.num_q_heads
        * self.q_seq_len
        * self.kv_seq_len
        * active_fraction
    )
    qk_bwd_flops = 2.0 * active_tokens * self.dim_qk
    pv_bwd_flops = 2.0 * active_tokens * self.dim_v
    dp_bwd_flops = 2.0 * active_tokens * self.dim_v
    dk_bwd_flops = 2.0 * active_tokens * self.dim_qk
    dq_bwd_flops = 2.0 * active_tokens * self.dim_qk
    return (
        qk_bwd_flops + pv_bwd_flops + dp_bwd_flops + dk_bwd_flops + dq_bwd_flops
    )


CANONICAL_BENCHMARK_CONFIGS = (
    SplashBwdConfig(
        case_name="mha_8k_h4",
        batch_size=1,
        q_seq_len=8192,
        kv_seq_len=8192,
        num_q_heads=4,
        num_kv_heads=4,
        dim_qk=128,
        dim_v=128,
        block_q=1024,
        block_kv=1024,
        block_q_head=4,
        block_kv_head=4,
        pl_block_q=8192,
        pl_block_kv=2048,
        pl_block_kv_compute=2048,
        pl_block_q_dkv=1024,
        pl_block_kv_dkv=2048,
        pl_block_kv_dkv_compute=512,
    ),
    SplashBwdConfig(
        case_name="gqa_8k_hq4_hkv1",
        batch_size=1,
        q_seq_len=8192,
        kv_seq_len=8192,
        num_q_heads=4,
        num_kv_heads=1,
        dim_qk=128,
        dim_v=128,
        block_q=1024,
        block_kv=1024,
        block_q_head=4,
        block_kv_head=1,
        pl_block_q=512,
        pl_block_kv=2048,
        pl_block_kv_compute=8192,
        pl_block_q_dkv=2048,
        pl_block_kv_dkv=2048,
        pl_block_kv_dkv_compute=256,
    ),
    SplashBwdConfig(
        case_name="mha_8k_h8",
        batch_size=1,
        q_seq_len=8192,
        kv_seq_len=8192,
        num_q_heads=8,
        num_kv_heads=8,
        dim_qk=128,
        dim_v=128,
        block_q=1024,
        block_kv=1024,
        block_q_head=8,
        block_kv_head=8,
        pl_block_q=8192,
        pl_block_kv=1024,
        pl_block_kv_compute=2048,
        pl_block_q_dkv=2048,
        pl_block_kv_dkv=2048,
        pl_block_kv_dkv_compute=256,
    ),
    SplashBwdConfig(
        case_name="gqa_8k_hq8_hkv1",
        batch_size=1,
        q_seq_len=8192,
        kv_seq_len=8192,
        num_q_heads=8,
        num_kv_heads=1,
        dim_qk=128,
        dim_v=128,
        block_q=1024,
        block_kv=1024,
        block_q_head=8,
        block_kv_head=1,
        pl_block_q=512,
        pl_block_kv=2048,
        pl_block_kv_compute=8192,
        pl_block_q_dkv=2048,
        pl_block_kv_dkv=2048,
        pl_block_kv_dkv_compute=256,
    ),
    SplashBwdConfig(
        case_name="gqa_16k_hq8_hkv1",
        batch_size=1,
        q_seq_len=16384,
        kv_seq_len=16384,
        num_q_heads=8,
        num_kv_heads=1,
        dim_qk=128,
        dim_v=128,
        block_q=1024,
        block_kv=1024,
        block_q_head=8,
        block_kv_head=1,
        pl_block_q=2048,
        pl_block_kv=256,
        pl_block_kv_compute=1024,
        pl_block_q_dkv=1024,
        pl_block_kv_dkv=2048,
        pl_block_kv_dkv_compute=512,
    ),
)
