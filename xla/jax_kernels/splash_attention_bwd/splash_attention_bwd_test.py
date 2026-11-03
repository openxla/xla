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

"""Tests for jax_k Splash Attention backward kernels."""

import functools
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from xla.jax_kernels import splash_attention_bwd
from xla.jax_kernels.splash_attention_bwd import champion
from xla.jax_kernels.splash_attention_bwd import configs
from xla.jax_kernels.splash_attention_bwd import reference as baseline

mask_lib = baseline
LOG2E = configs.LOG2E
SegmentIds = champion.SegmentIds


class JaxKSplashAttentionBwdTest(parameterized.TestCase):

  def _assert_allclose(self, actual, expected, atol=0.01, rtol=0.01):
    self.assertTrue(np.all(np.isfinite(np.asarray(actual))))
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol
    )

  def _make_qkv_do(
      self,
      key: jax.Array,
      *,
      batch_size: int,
      num_q_heads: int,
      num_kv_heads: int,
      q_seq_len: int,
      kv_seq_len: int,
      head_dim: int,
      dtype: Any,
      is_mqa: bool = False
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    k1, k2, k3, k4 = random.split(key, 4)
    q = random.uniform(
        k1, (batch_size, num_q_heads, q_seq_len, head_dim), dtype=dtype
    )
    if is_mqa:
      k = random.uniform(k2, (batch_size, kv_seq_len, head_dim), dtype=dtype)
      v = random.uniform(k3, (batch_size, kv_seq_len, head_dim), dtype=dtype)
    else:
      k = random.uniform(
          k2, (batch_size, num_kv_heads, kv_seq_len, head_dim), dtype=dtype
      )
      v = random.uniform(
          k3, (batch_size, num_kv_heads, kv_seq_len, head_dim), dtype=dtype
      )
    do = random.uniform(
        k4, (batch_size, num_q_heads, q_seq_len, head_dim), dtype=dtype
    )
    return (q, k, v, do)

  @parameterized.named_parameters(
      dict(
          testcase_name='mha_causal_f32',
          num_q_heads=4,
          num_kv_heads=4,
          is_mqa=False,
          is_causal=True,
          dtype=jnp.float32,
          soft_cap=None,
      ),
      dict(
          testcase_name='mha_full_softcap_f32',
          num_q_heads=4,
          num_kv_heads=4,
          is_mqa=False,
          is_causal=False,
          dtype=jnp.float32,
          soft_cap=20.0,
      ),
      dict(
          testcase_name='gqa_causal_f32',
          num_q_heads=8,
          num_kv_heads=2,
          is_mqa=False,
          is_causal=True,
          dtype=jnp.float32,
          soft_cap=None,
      ),
      dict(
          testcase_name='mqa_causal_f32',
          num_q_heads=8,
          num_kv_heads=1,
          is_mqa=True,
          is_causal=True,
          dtype=jnp.float32,
          soft_cap=None,
      ),
      dict(
          testcase_name='mha_causal_bf16',
          num_q_heads=8,
          num_kv_heads=8,
          is_mqa=False,
          is_causal=True,
          dtype=jnp.bfloat16,
          soft_cap=None,
      ),
      dict(
          testcase_name='gqa_causal_bf16',
          num_q_heads=8,
          num_kv_heads=1,
          is_mqa=False,
          is_causal=True,
          dtype=jnp.bfloat16,
          soft_cap=None,
      ),
  )
  def test_jax_k_bwd_correctness(
      self,
      num_q_heads: int,
      num_kv_heads: int,
      is_mqa: bool,
      is_causal: bool,
      dtype: Any,
      soft_cap: float | None,
  ):
    """Verifies numerical accuracy of jax_k backward kernel against reference."""
    batch_size = 1
    q_seq_len = 1024
    kv_seq_len = 1024
    head_dim = 128
    q, k, v, do = self._make_qkv_do(
        random.key(42),
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        head_dim=head_dim,
        dtype=dtype,
        is_mqa=is_mqa,
    )
    mask_2d = (
        mask_lib.CausalMask((q_seq_len, kv_seq_len))
        if is_causal
        else mask_lib.FullMask((q_seq_len, kv_seq_len))
    )
    multi_mask = mask_lib.MultiHeadMask(
        tuple(mask_2d for _ in range(num_q_heads))
    )
    mask_arr = jnp.array(multi_mask[:, :, :])
    q32, k32, v32, do32 = jax.tree.map(
        lambda x: x.astype(jnp.float32), (q, k, v, do)
    )
    o_ref, logsumexp = jax.block_until_ready(
        baseline.reference_splash_attention_fwd(
            multi_mask,
            q32,
            k32,
            v32,
            sinks=None,
            is_mqa=is_mqa,
            attn_logits_soft_cap=soft_cap,
        )
    )
    dq_ref, dk_ref, dv_ref, _ = jax.block_until_ready(
        baseline.reference_splash_attention_bwd(
            mask_arr,
            q32,
            k32,
            v32,
            None,
            None,
            o_ref,
            logsumexp,
            do32,
            num_q_heads,
            num_kv_heads,
            soft_cap,
            is_mqa,
        )
    )
    dq_jax_k, dk_jax_k, dv_jax_k = jax.block_until_ready(
        splash_attention_bwd.splash_attention_bwd_jax_k(
            q=q,
            k=k,
            v=v,
            o=o_ref.astype(dtype),
            do=do,
            logsumexp=logsumexp,
            mask=None if is_causal else mask_arr,
            block_q=512,
            block_kv=512,
            block_q_head=min(num_q_heads // num_kv_heads, 4)
            if is_mqa or num_kv_heads < num_q_heads
            else num_q_heads,
            block_kv_head=1
            if is_mqa or num_kv_heads < num_q_heads
            else num_kv_heads,
            is_mqa=is_mqa,
            is_causal=is_causal,
            use_log2=False,
            attn_logits_soft_cap=soft_cap,
            dtype=dtype,
        )
    )
    if dtype == jnp.float32:
      self._assert_allclose(
          dv_jax_k, dv_ref.astype(dtype), atol=0.01, rtol=0.01
      )
      self._assert_allclose(
          dq_jax_k, dq_ref.astype(dtype), atol=0.01, rtol=0.01
      )
      self._assert_allclose(
          dk_jax_k, dk_ref.astype(dtype), atol=0.01, rtol=0.01
      )
    else:
      self._assert_allclose(dv_jax_k, dv_ref.astype(dtype), atol=0.2, rtol=0.1)
      self._assert_allclose(dq_jax_k, dq_ref.astype(dtype), atol=0.3, rtol=0.2)
      self._assert_allclose(dk_jax_k, dk_ref.astype(dtype), atol=0.3, rtol=0.2)

  @parameterized.named_parameters(
      dict(
          testcase_name='mha_causal_seg_b2_ne_h4',
          batch_size=2,
          num_q_heads=4,
          num_kv_heads=4,
          is_causal=True,
          pass_mask=False,
      ),
      dict(
          testcase_name='mha_causal_seg_b4_eq_h4',
          batch_size=4,
          num_q_heads=4,
          num_kv_heads=4,
          is_causal=True,
          pass_mask=True,
      ),
      dict(
          testcase_name='gqa_causal_seg_b2_hq4_hkv1',
          batch_size=2,
          num_q_heads=4,
          num_kv_heads=1,
          is_causal=True,
          pass_mask=False,
      ),
      dict(
          testcase_name='mha_full_seg_b2',
          batch_size=2,
          num_q_heads=4,
          num_kv_heads=4,
          is_causal=False,
          pass_mask=False,
      ),
  )
  def test_jax_k_bwd_with_segment_ids(
      self,
      batch_size: int,
      num_q_heads: int,
      num_kv_heads: int,
      is_causal: bool,
      pass_mask: bool,
  ):
    """Verifies segment ID masking with causal peeling and multi batch inputs."""
    q_seq_len = 1024
    kv_seq_len = 1024
    head_dim = 128
    dtype = jnp.float32
    q, k, v, do = self._make_qkv_do(
        random.key(123),
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        head_dim=head_dim,
        dtype=dtype,
    )
    seg_rows = []
    for b in range(batch_size):
      num_docs = 2 if b % 2 == 0 else 4
      doc_len = q_seq_len // num_docs
      seg_rows.append(
          jnp.repeat(jnp.arange(num_docs, dtype=jnp.int32) + b * 10, doc_len)
      )
    q_seg = jnp.stack(seg_rows, axis=0)
    segment_ids = SegmentIds(q=q_seg, kv=q_seg)
    mask_2d = (
        mask_lib.CausalMask((q_seq_len, kv_seq_len))
        if is_causal
        else mask_lib.FullMask((q_seq_len, kv_seq_len))
    )
    multi_mask = mask_lib.MultiHeadMask(
        tuple(mask_2d for _ in range(num_q_heads))
    )
    mask_arr = jnp.array(multi_mask[:, :, :])
    o_ref, logsumexp = jax.block_until_ready(
        baseline.reference_splash_attention_fwd(
            multi_mask,
            q,
            k,
            v,
            segment_ids=segment_ids,
            sinks=None,
            is_mqa=False,
            attn_logits_soft_cap=None,
        )
    )
    dq_ref, dk_ref, dv_ref, _ = jax.block_until_ready(
        baseline.reference_splash_attention_bwd(
            mask_arr,
            q,
            k,
            v,
            segment_ids,
            None,
            o_ref,
            logsumexp,
            do,
            num_q_heads,
            num_kv_heads,
            None,
            False,
        )
    )
    dq_jax_k, dk_jax_k, dv_jax_k = jax.block_until_ready(
        champion.splash_attention_bwd_jax_k(
            q=q,
            k=k,
            v=v,
            o=o_ref,
            do=do,
            logsumexp=logsumexp * LOG2E,
            segment_ids=segment_ids,
            mask=mask_arr if pass_mask else None,
            block_q=512,
            block_kv=512,
            is_mqa=False,
            is_causal=is_causal,
            use_log2=True,
            dtype=dtype,
        )
    )
    self._assert_allclose(dv_jax_k, dv_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dq_jax_k, dq_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dk_jax_k, dk_ref, atol=0.01, rtol=0.01)

  @parameterized.named_parameters(
      dict(
          testcase_name='mha_no_softcap_f32',
          num_q_heads=4,
          num_kv_heads=4,
          soft_cap=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name='mha_softcap_f32',
          num_q_heads=4,
          num_kv_heads=4,
          soft_cap=15.0,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name='gqa_no_softcap_f32',
          num_q_heads=4,
          num_kv_heads=1,
          soft_cap=None,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name='gqa_softcap_f32',
          num_q_heads=4,
          num_kv_heads=1,
          soft_cap=15.0,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name='gqa_no_softcap_bf16',
          num_q_heads=4,
          num_kv_heads=1,
          soft_cap=None,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name='gqa_softcap_bf16',
          num_q_heads=4,
          num_kv_heads=1,
          soft_cap=15.0,
          dtype=jnp.bfloat16,
      ),
  )
  def test_jax_k_bwd_use_log2_parity(
      self,
      num_q_heads: int,
      num_kv_heads: int,
      soft_cap: float | None,
      dtype: Any,
  ):
    """Verifies base 2 and natural exponentiation parity with non zero logsumexp."""
    batch_size = 1
    q_seq_len = 1024
    kv_seq_len = 1024
    head_dim = 128
    q, k, v, do = self._make_qkv_do(
        random.key(77),
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        head_dim=head_dim,
        dtype=dtype,
    )
    mask_2d = mask_lib.CausalMask((q_seq_len, kv_seq_len))
    multi_mask = mask_lib.MultiHeadMask(
        tuple(mask_2d for _ in range(num_q_heads))
    )
    q32, k32, v32, do32 = jax.tree.map(
        lambda x: x.astype(jnp.float32), (q, k, v, do)
    )
    o_ref, logsumexp = jax.block_until_ready(
        baseline.reference_splash_attention_fwd(
            multi_mask, q32, k32, v32, attn_logits_soft_cap=soft_cap
        )
    )
    dq_ref, dk_ref, dv_ref, _ = jax.block_until_ready(
        baseline.reference_splash_attention_bwd(
            multi_mask,
            q32,
            k32,
            v32,
            None,
            None,
            o_ref,
            logsumexp,
            do32,
            num_q_heads,
            num_kv_heads,
            soft_cap,
            False,
        )
    )
    dq_natural, dk_natural, dv_natural = jax.block_until_ready(
        champion.splash_attention_bwd_jax_k(
            q=q,
            k=k,
            v=v,
            o=o_ref.astype(dtype),
            do=do,
            logsumexp=logsumexp.astype(dtype),
            block_q=512,
            block_kv=512,
            is_mqa=False,
            is_causal=True,
            use_log2=False,
            attn_logits_soft_cap=soft_cap,
            dtype=dtype,
        )
    )
    dq_base2, dk_base2, dv_base2 = jax.block_until_ready(
        champion.splash_attention_bwd_jax_k(
            q=q,
            k=k,
            v=v,
            o=o_ref.astype(dtype),
            do=do,
            logsumexp=(logsumexp * LOG2E).astype(dtype),
            block_q=512,
            block_kv=512,
            is_mqa=False,
            is_causal=True,
            use_log2=True,
            attn_logits_soft_cap=soft_cap,
            dtype=dtype,
        )
    )
    if dtype == jnp.float32:
      self._assert_allclose(dv_base2, dv_natural, atol=0.01, rtol=0.01)
      self._assert_allclose(dq_base2, dq_natural, atol=0.01, rtol=0.01)
      self._assert_allclose(dk_base2, dk_natural, atol=0.01, rtol=0.01)
      self._assert_allclose(dv_base2, dv_ref, atol=0.01, rtol=0.01)
      self._assert_allclose(dq_base2, dq_ref, atol=0.01, rtol=0.01)
      self._assert_allclose(dk_base2, dk_ref, atol=0.01, rtol=0.01)
    else:
      self._assert_allclose(
          dv_base2, dv_ref.astype(dtype), atol=0.25, rtol=0.15
      )
      self._assert_allclose(dq_base2, dq_ref.astype(dtype), atol=0.35, rtol=0.2)
      self._assert_allclose(dk_base2, dk_ref.astype(dtype), atol=0.35, rtol=0.2)

  def test_jax_k_bwd_causal_with_local_and_multihead_masks(self):
    """Verifies causal peeling combined with user masks and per head masks."""
    batch_size = 1
    num_q_heads = 4
    q_seq_len = 1024
    kv_seq_len = 1024
    head_dim = 128
    dtype = jnp.float32
    for num_kv_heads in (4, 1):
      q, k, v, do = self._make_qkv_do(
          random.key(314 + num_kv_heads),
          batch_size=batch_size,
          num_q_heads=num_q_heads,
          num_kv_heads=num_kv_heads,
          q_seq_len=q_seq_len,
          kv_seq_len=kv_seq_len,
          head_dim=head_dim,
          dtype=dtype,
      )
      local_2d = mask_lib.LocalMask(
          (q_seq_len, kv_seq_len), window_size=(256, 256), offset=0
      )
      causal_2d = mask_lib.CausalMask((q_seq_len, kv_seq_len))
      combined_2d = causal_2d & local_2d
      local_arr = jnp.asarray(local_2d[:, :], dtype=jnp.bool_)
      o_ref, lse_ref = baseline.reference_splash_attention_fwd(
          combined_2d, q, k, v
      )
      dq_ref, dk_ref, dv_ref, _ = baseline.reference_splash_attention_bwd(
          combined_2d,
          q,
          k,
          v,
          None,
          None,
          o_ref,
          lse_ref,
          do,
          num_q_heads,
          num_kv_heads,
      )
      dq_jax_k, dk_jax_k, dv_jax_k = jax.block_until_ready(
          champion.splash_attention_bwd_jax_k(
              q=q,
              k=k,
              v=v,
              o=o_ref,
              do=do,
              logsumexp=lse_ref * LOG2E,
              mask=local_arr,
              block_q=512,
              block_kv=512,
              is_causal=True,
              use_log2=True,
              dtype=dtype,
          )
      )
      self._assert_allclose(dv_jax_k, dv_ref, atol=0.01, rtol=0.01)
      self._assert_allclose(dq_jax_k, dq_ref, atol=0.01, rtol=0.01)
      self._assert_allclose(dk_jax_k, dk_ref, atol=0.01, rtol=0.01)

  @parameterized.named_parameters(
      dict(
          testcase_name='rect_mha_512x1024',
          num_q_heads=4,
          num_kv_heads=4,
          q_seq_len=512,
          kv_seq_len=1024,
          block_q=512,
          block_kv=512,
          block_q_head=4,
          block_kv_head=4,
      ),
      dict(
          testcase_name='rect_gqa_512x1024',
          num_q_heads=4,
          num_kv_heads=1,
          q_seq_len=512,
          kv_seq_len=1024,
          block_q=512,
          block_kv=512,
          block_q_head=4,
          block_kv_head=1,
      ),
      dict(
          testcase_name='unequal_blocks_mha_256x512',
          num_q_heads=4,
          num_kv_heads=4,
          q_seq_len=1024,
          kv_seq_len=1024,
          block_q=256,
          block_kv=512,
          block_q_head=4,
          block_kv_head=4,
      ),
      dict(
          testcase_name='unequal_blocks_gqa_256x512',
          num_q_heads=4,
          num_kv_heads=1,
          q_seq_len=1024,
          kv_seq_len=1024,
          block_q=256,
          block_kv=512,
          block_q_head=4,
          block_kv_head=1,
      ),
      dict(
          testcase_name='gqa_multi_kv_head_tile',
          num_q_heads=8,
          num_kv_heads=4,
          q_seq_len=1024,
          kv_seq_len=1024,
          block_q=512,
          block_kv=512,
          block_q_head=2,
          block_kv_head=2,
      ),
  )
  def test_jax_k_bwd_rectangular_unequal_blocks_and_multi_kv_head(
      self,
      num_q_heads: int,
      num_kv_heads: int,
      q_seq_len: int,
      kv_seq_len: int,
      block_q: int,
      block_kv: int,
      block_q_head: int,
      block_kv_head: int,
  ):
    """Verifies rectangular causal, unequal block sizes, and multi KV head GQA."""
    dtype = jnp.float32
    q, k, v, do = self._make_qkv_do(
        random.key(555),
        batch_size=1,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        head_dim=128,
        dtype=dtype,
    )
    causal_2d = mask_lib.CausalMask((q_seq_len, kv_seq_len))
    o_ref, lse_ref = baseline.reference_splash_attention_fwd(causal_2d, q, k, v)
    dq_ref, dk_ref, dv_ref, _ = baseline.reference_splash_attention_bwd(
        causal_2d,
        q,
        k,
        v,
        None,
        None,
        o_ref,
        lse_ref,
        do,
        num_q_heads,
        num_kv_heads,
    )
    dq_jax_k, dk_jax_k, dv_jax_k = jax.block_until_ready(
        champion.splash_attention_bwd_jax_k(
            q=q,
            k=k,
            v=v,
            o=o_ref,
            do=do,
            logsumexp=lse_ref * LOG2E,
            block_q=block_q,
            block_kv=block_kv,
            block_q_head=block_q_head,
            block_kv_head=block_kv_head,
            is_causal=True,
            use_log2=True,
            dtype=dtype,
        )
    )
    self._assert_allclose(dv_jax_k, dv_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dq_jax_k, dq_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dk_jax_k, dk_ref, atol=0.01, rtol=0.01)

  @parameterized.named_parameters(
      dict(
          testcase_name='mha_s1280_b512_causal',
          num_q_heads=4,
          num_kv_heads=4,
          seq_len=1280,
          block_size=512,
          is_causal=True,
      ),
      dict(
          testcase_name='gqa_s1280_b512_causal',
          num_q_heads=4,
          num_kv_heads=1,
          seq_len=1280,
          block_size=512,
          is_causal=True,
      ),
      dict(
          testcase_name='mha_s1536_b1024_full',
          num_q_heads=4,
          num_kv_heads=4,
          seq_len=1536,
          block_size=1024,
          is_causal=False,
      ),
  )
  def test_jax_k_bwd_non_divisible_seq_len(
      self,
      num_q_heads: int,
      num_kv_heads: int,
      seq_len: int,
      block_size: int,
      is_causal: bool,
  ):
    """Verifies exact tail gradients when sequence length is not a tile multiple."""
    dtype = jnp.float32
    q, k, v, do = self._make_qkv_do(
        random.key(888),
        batch_size=1,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
        head_dim=128,
        dtype=dtype,
    )
    mask_2d = (
        mask_lib.CausalMask((seq_len, seq_len))
        if is_causal
        else mask_lib.FullMask((seq_len, seq_len))
    )
    o_ref, lse_ref = baseline.reference_splash_attention_fwd(mask_2d, q, k, v)
    dq_ref, dk_ref, dv_ref, _ = baseline.reference_splash_attention_bwd(
        mask_2d,
        q,
        k,
        v,
        None,
        None,
        o_ref,
        lse_ref,
        do,
        num_q_heads,
        num_kv_heads,
    )
    dq_jax_k, dk_jax_k, dv_jax_k = jax.block_until_ready(
        champion.splash_attention_bwd_jax_k(
            q=q,
            k=k,
            v=v,
            o=o_ref,
            do=do,
            logsumexp=lse_ref * LOG2E,
            block_q=block_size,
            block_kv=block_size,
            is_causal=is_causal,
            use_log2=True,
            dtype=dtype,
        )
    )
    self._assert_allclose(dv_jax_k, dv_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dq_jax_k, dq_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dk_jax_k, dk_ref, atol=0.01, rtol=0.01)
    with self.assertRaises(ValueError):
      champion.splash_attention_bwd_jax_k(
          q=q,
          k=k,
          v=v,
          o=o_ref,
          do=do,
          logsumexp=lse_ref,
          block_q=0,
          block_kv=512,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='mha_4d_singleton_head',
          num_q_heads=4,
          num_kv_heads=4,
          block_q_head=4,
          block_kv_head=4,
      ),
      dict(
          testcase_name='gqa_3d_and_4d_singleton_head',
          num_q_heads=4,
          num_kv_heads=1,
          block_q_head=4,
          block_kv_head=1,
      ),
      dict(
          testcase_name='gqa_multikvh_singleton_head',
          num_q_heads=8,
          num_kv_heads=4,
          block_q_head=2,
          block_kv_head=2,
      ),
  )
  def test_jax_k_bwd_singleton_head_and_batch_broadcast_masks(
      self,
      num_q_heads: int,
      num_kv_heads: int,
      block_q_head: int,
      block_kv_head: int,
  ):
    """Verifies broadcastable (1, Sq, Skv) and (B, 1, Sq, Skv) masks."""
    batch_size = 2
    seq_len = 1024
    head_dim = 128
    dtype = jnp.float32
    q, k, v, do = self._make_qkv_do(
        random.key(611 + num_kv_heads),
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype,
    )
    m0 = mask_lib.LocalMask((seq_len, seq_len), window_size=(256, 0), offset=0)
    m1 = mask_lib.ChunkedCausalMask((seq_len, seq_len), chunk_size=256)
    mask_b0 = jnp.asarray(m0[:, :], dtype=jnp.bool_)
    mask_b1 = jnp.asarray(m1[:, :], dtype=jnp.bool_)
    mask_4d_b1 = jnp.stack([mask_b0[None], mask_b1[None]], axis=0)
    o0, lse0 = baseline.reference_splash_attention_fwd(
        m0, q[0:1], k[0:1], v[0:1]
    )
    dq0, dk0, dv0, _ = baseline.reference_splash_attention_bwd(
        m0,
        q[0:1],
        k[0:1],
        v[0:1],
        None,
        None,
        o0,
        lse0,
        do[0:1],
        num_q_heads,
        num_kv_heads,
    )
    o1, lse1 = baseline.reference_splash_attention_fwd(
        m1, q[1:2], k[1:2], v[1:2]
    )
    dq1, dk1, dv1, _ = baseline.reference_splash_attention_bwd(
        m1,
        q[1:2],
        k[1:2],
        v[1:2],
        None,
        None,
        o1,
        lse1,
        do[1:2],
        num_q_heads,
        num_kv_heads,
    )
    o_ref = jnp.concatenate([o0, o1], axis=0)
    lse_ref = jnp.concatenate([lse0, lse1], axis=0)
    dq_ref = jnp.concatenate([dq0, dq1], axis=0)
    dk_ref = jnp.concatenate([dk0, dk1], axis=0)
    dv_ref = jnp.concatenate([dv0, dv1], axis=0)
    dq_jax_k, dk_jax_k, dv_jax_k = jax.block_until_ready(
        champion.splash_attention_bwd_jax_k(
            q=q,
            k=k,
            v=v,
            o=o_ref,
            do=do,
            logsumexp=lse_ref * LOG2E,
            mask=mask_4d_b1,
            block_q=512,
            block_kv=512,
            block_q_head=block_q_head,
            block_kv_head=block_kv_head,
            is_causal=False,
            use_log2=True,
            dtype=dtype,
        )
    )
    self._assert_allclose(dq_jax_k, dq_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dk_jax_k, dk_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dv_jax_k, dv_ref, atol=0.01, rtol=0.01)
    mask_3d_1 = mask_b0[None, :, :]
    dq_3d, dk_3d, dv_3d = jax.block_until_ready(
        champion.splash_attention_bwd_jax_k(
            q=q[0:1],
            k=k[0:1],
            v=v[0:1],
            o=o0,
            do=do[0:1],
            logsumexp=lse0 * LOG2E,
            mask=mask_3d_1,
            block_q=512,
            block_kv=512,
            block_q_head=block_q_head,
            block_kv_head=block_kv_head,
            is_causal=False,
            use_log2=True,
            dtype=dtype,
        )
    )
    self._assert_allclose(dq_3d, dq0, atol=0.01, rtol=0.01)
    self._assert_allclose(dk_3d, dk0, atol=0.01, rtol=0.01)
    self._assert_allclose(dv_3d, dv0, atol=0.01, rtol=0.01)
    with self.assertRaises(ValueError):
      champion.splash_attention_bwd_jax_k(
          q=q,
          k=k,
          v=v,
          o=o_ref,
          do=do,
          logsumexp=lse_ref,
          mask=jnp.ones((batch_size, 3, seq_len, seq_len), dtype=jnp.bool_),
          block_q=512,
          block_kv=512,
          is_causal=False,
      )

  @parameterized.named_parameters(
      dict(testcase_name='mha', num_q_heads=4, num_kv_heads=4),
      dict(testcase_name='gqa', num_q_heads=4, num_kv_heads=1),
  )
  def test_jax_k_bwd_direct_vmap_and_vmap_jit(
      self, num_q_heads: int, num_kv_heads: int
  ):
    """Verifies jax.vmap(splash_attention_bwd_jax_k) with batch >= 2."""
    batch_size = 2
    seq_len = 1024
    head_dim = 128
    dtype = jnp.float32
    q, k, v, do = self._make_qkv_do(
        random.key(711 + num_kv_heads),
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype,
    )
    causal_2d = mask_lib.CausalMask((seq_len, seq_len))
    o_ref, lse_ref = baseline.reference_splash_attention_fwd(causal_2d, q, k, v)
    dq_ref, dk_ref, dv_ref, _ = baseline.reference_splash_attention_bwd(
        causal_2d,
        q,
        k,
        v,
        None,
        None,
        o_ref,
        lse_ref,
        do,
        num_q_heads,
        num_kv_heads,
    )
    bwd_fn = functools.partial(
        champion.splash_attention_bwd_jax_k,
        block_q=512,
        block_kv=512,
        is_causal=True,
        use_log2=True,
        dtype=dtype,
    )
    dq_vmap, dk_vmap, dv_vmap = jax.block_until_ready(
        jax.vmap(bwd_fn)(q, k, v, o_ref, do, lse_ref * LOG2E)
    )
    self._assert_allclose(dq_vmap, dq_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dk_vmap, dk_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dv_vmap, dv_ref, atol=0.01, rtol=0.01)
    dq_vjit, dk_vjit, dv_vjit = jax.block_until_ready(
        jax.vmap(jax.jit(bwd_fn))(q, k, v, o_ref, do, lse_ref * LOG2E)
    )
    self._assert_allclose(dq_vjit, dq_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dk_vjit, dk_ref, atol=0.01, rtol=0.01)
    self._assert_allclose(dv_vjit, dv_ref, atol=0.01, rtol=0.01)

  @parameterized.named_parameters(
      dict(testcase_name='mha', num_q_heads=4, num_kv_heads=4),
      dict(testcase_name='gqa', num_q_heads=4, num_kv_heads=1),
  )
  def test_jax_k_bwd_neginf_logsumexp_on_all_masked_rows(
      self, num_q_heads: int, num_kv_heads: int
  ):
    """Verifies logsumexp = -inf on all masked padding rows produces no NaN."""
    seq_len = 1024
    valid_len = 768
    head_dim = 128
    dtype = jnp.float32
    q, k, v, do = self._make_qkv_do(
        random.key(811 + num_kv_heads),
        batch_size=1,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype,
    )
    pad_mask_q = (jnp.arange(seq_len) < valid_len)[None, None, :, None]
    do_masked = jnp.where(pad_mask_q, do, 0.0)
    q_seg = jnp.where(jnp.arange(seq_len) < valid_len, 0, -1).astype(jnp.int32)[
        None, :
    ]
    kv_seg = jnp.where(jnp.arange(seq_len) < valid_len, 0, -2).astype(
        jnp.int32
    )[None, :]
    seg_ids = SegmentIds(q=q_seg, kv=kv_seg)
    causal_valid = mask_lib.CausalMask((valid_len, valid_len))
    o_valid, lse_valid = baseline.reference_splash_attention_fwd(
        causal_valid,
        q[:, :, :valid_len, :],
        k[:, :, :valid_len, :],
        v[:, :, :valid_len, :],
    )
    dq_valid, dk_valid, dv_valid, _ = baseline.reference_splash_attention_bwd(
        causal_valid,
        q[:, :, :valid_len, :],
        k[:, :, :valid_len, :],
        v[:, :, :valid_len, :],
        None,
        None,
        o_valid,
        lse_valid,
        do[:, :, :valid_len, :],
        num_q_heads,
        num_kv_heads,
    )
    o_padded = jnp.pad(
        o_valid, ((0, 0), (0, 0), (0, seq_len - valid_len), (0, 0))
    )
    lse_padded = jnp.pad(
        lse_valid * LOG2E,
        ((0, 0), (0, 0), (0, seq_len - valid_len)),
        constant_values=-jnp.inf,
    )
    dq_jax_k, dk_jax_k, dv_jax_k = jax.block_until_ready(
        champion.splash_attention_bwd_jax_k(
            q=q,
            k=k,
            v=v,
            o=o_padded,
            do=do_masked,
            logsumexp=lse_padded,
            segment_ids=seg_ids,
            block_q=512,
            block_kv=512,
            is_causal=True,
            use_log2=True,
            dtype=dtype,
        )
    )
    self._assert_allclose(
        dq_jax_k[:, :, :valid_len, :], dq_valid, atol=0.01, rtol=0.01
    )
    self._assert_allclose(
        dk_jax_k[:, :, :valid_len, :], dk_valid, atol=0.01, rtol=0.01
    )
    self._assert_allclose(
        dv_jax_k[:, :, :valid_len, :], dv_valid, atol=0.01, rtol=0.01
    )
    self._assert_allclose(
        dq_jax_k[:, :, valid_len:, :],
        jnp.zeros_like(dq_jax_k[:, :, valid_len:, :]),
        atol=1e-06,
        rtol=1e-06,
    )
    self._assert_allclose(
        dk_jax_k[:, :, valid_len:, :],
        jnp.zeros_like(dk_jax_k[:, :, valid_len:, :]),
        atol=1e-06,
        rtol=1e-06,
    )
    self._assert_allclose(
        dv_jax_k[:, :, valid_len:, :],
        jnp.zeros_like(dv_jax_k[:, :, valid_len:, :]),
        atol=1e-06,
        rtol=1e-06,
    )

  @parameterized.named_parameters(
      dict(testcase_name='mha_8k_bf16', num_q_heads=4, num_kv_heads=4),
      dict(testcase_name='gqa_8k_bf16', num_q_heads=4, num_kv_heads=1),
  )
  def test_jax_k_bwd_bf16_long_sequence_precision(
      self, num_q_heads: int, num_kv_heads: int
  ):
    """Verifies 8k bfloat16 relative Frobenius error stays under 1% vs f32 ref."""
    seq_len = 8192
    head_dim = 128
    scale = float(head_dim) ** (-0.5)
    k1, k2, k3, k4 = random.split(random.key(911 + num_kv_heads), 4)
    q = random.normal(
        k1, (1, num_q_heads, seq_len, head_dim), dtype=jnp.bfloat16
    ) * jnp.asarray(scale, dtype=jnp.bfloat16)
    k = random.normal(
        k2, (1, num_kv_heads, seq_len, head_dim), dtype=jnp.bfloat16
    ) * jnp.asarray(scale, dtype=jnp.bfloat16)
    v = random.normal(
        k3, (1, num_kv_heads, seq_len, head_dim), dtype=jnp.bfloat16
    ) * jnp.asarray(scale, dtype=jnp.bfloat16)
    do = random.normal(
        k4, (1, num_q_heads, seq_len, head_dim), dtype=jnp.bfloat16
    )
    causal_2d = mask_lib.CausalMask((seq_len, seq_len))
    q32, k32, v32, do32 = jax.tree.map(
        lambda x: x.astype(jnp.float32), (q, k, v, do)
    )
    o_ref, lse_ref = baseline.reference_splash_attention_fwd(
        causal_2d, q32, k32, v32
    )
    dq_ref, dk_ref, dv_ref, _ = baseline.reference_splash_attention_bwd(
        causal_2d,
        q32,
        k32,
        v32,
        None,
        None,
        o_ref,
        lse_ref,
        do32,
        num_q_heads,
        num_kv_heads,
    )
    dq_jax_k, dk_jax_k, dv_jax_k = jax.block_until_ready(
        champion.splash_attention_bwd_jax_k(
            q=q,
            k=k,
            v=v,
            o=o_ref.astype(jnp.bfloat16),
            do=do,
            logsumexp=lse_ref * LOG2E,
            block_q=1024,
            block_kv=1024,
            is_causal=True,
            use_log2=True,
            dtype=jnp.bfloat16,
        )
    )

    def _rel_fro(actual: jax.Array, ref: jax.Array) -> float:
      diff = np.asarray(actual, dtype=np.float32) - np.asarray(
          ref, dtype=np.float32
      )
      return float(
          np.linalg.norm(diff)
          / np.maximum(np.linalg.norm(np.asarray(ref, dtype=np.float32)), 1e-12)
      )

    self.assertLess(_rel_fro(dq_jax_k, dq_ref), 0.01)
    self.assertLess(_rel_fro(dk_jax_k, dk_ref), 0.01)
    self.assertLess(_rel_fro(dv_jax_k, dv_ref), 0.01)

  @parameterized.named_parameters(
      dict(
          testcase_name='mha_with_segment_ids_f32',
          num_kv_heads=4,
          with_segment_ids=True,
          seq_len=2048,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name='mha_without_segment_ids_bf16',
          num_kv_heads=4,
          with_segment_ids=False,
          seq_len=2048,
          dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name='gqa_with_segment_ids_f32',
          num_kv_heads=1,
          with_segment_ids=True,
          seq_len=2048,
          dtype=jnp.float32,
      ),
      dict(
          testcase_name='mha_single_tile_512_f32',
          num_kv_heads=4,
          with_segment_ids=False,
          seq_len=512,
          dtype=jnp.float32,
      ),
  )
  def test_two_layer_scanned_attention_bwd(
      self, num_kv_heads: int, with_segment_ids: bool, seq_len: int, dtype: Any
  ):
    """Verifies two attention layers inside a single lax.scan backward body."""
    block_size = 512
    num_q_heads = 4
    head_dim = 128
    hidden_dim = num_q_heads * head_dim
    num_superblocks = 2
    attn_scale = float(head_dim) ** (-0.5)
    proj_scale = float(hidden_dim) ** (-0.5)
    k0, k1, k2, k3, k4 = random.split(
        random.key(1024 + int(with_segment_ids) + seq_len), 5
    )
    x0 = random.normal(k0, (1, seq_len, hidden_dim), dtype=dtype) * 0.2
    wq = (
        random.normal(
            k1,
            (num_superblocks, 2, hidden_dim, num_q_heads, head_dim),
            dtype=dtype,
        )
        * proj_scale
    )
    wk = (
        random.normal(
            k2,
            (num_superblocks, 2, hidden_dim, num_kv_heads, head_dim),
            dtype=dtype,
        )
        * proj_scale
    )
    wv = (
        random.normal(
            k3,
            (num_superblocks, 2, hidden_dim, num_kv_heads, head_dim),
            dtype=dtype,
        )
        * proj_scale
    )
    wo = (
        random.normal(
            k4,
            (num_superblocks, 2, num_q_heads, head_dim, hidden_dim),
            dtype=dtype,
        )
        * proj_scale
    )
    seg = (
        jnp.repeat(jnp.arange(4, dtype=jnp.int32), seq_len // 4)[None, :]
        if with_segment_ids
        else None
    )
    causal_mask = mask_lib.CausalMask((seq_len, seq_len))

    def _run_ref_4d_fwd(q_4d, k_4d, v_4d):
      ref_seg = SegmentIds(q=seg, kv=seg) if seg is not None else None
      o_f32, lse_4d = baseline.reference_splash_attention_fwd(
          causal_mask, q_4d, k_4d, v_4d, segment_ids=ref_seg
      )
      return (o_f32.astype(q_4d.dtype), lse_4d)

    @jax.custom_vjp
    def _ref_4d_attn(q_4d, k_4d, v_4d):
      o_4d, _ = _run_ref_4d_fwd(q_4d, k_4d, v_4d)
      return o_4d

    def _ref_4d_fwd(q_4d, k_4d, v_4d):
      o_4d, lse_4d = _run_ref_4d_fwd(q_4d, k_4d, v_4d)
      return (o_4d, (q_4d, k_4d, v_4d, o_4d, lse_4d))

    def _ref_4d_bwd(res_4d, do_4d):
      q_4d, k_4d, v_4d, o_4d, lse_4d = res_4d
      ref_seg = SegmentIds(q=seg, kv=seg) if seg is not None else None
      dq_r, dk_r, dv_r, _ = baseline.reference_splash_attention_bwd(
          causal_mask,
          q_4d,
          k_4d,
          v_4d,
          ref_seg,
          None,
          o_4d,
          lse_4d,
          do_4d,
          num_q_heads,
          num_kv_heads,
      )
      return (
          dq_r.astype(q_4d.dtype),
          dk_r.astype(k_4d.dtype),
          dv_r.astype(v_4d.dtype),
      )

    _ref_4d_attn.defvjp(_ref_4d_fwd, _ref_4d_bwd)

    @jax.custom_vjp
    def _direct_4d_attn(q_4d, k_4d, v_4d):
      o_4d, _ = _run_ref_4d_fwd(q_4d, k_4d, v_4d)
      return o_4d

    def _direct_4d_fwd(q_4d, k_4d, v_4d):
      o_4d, lse_4d = _run_ref_4d_fwd(q_4d, k_4d, v_4d)
      return (o_4d, (q_4d, k_4d, v_4d, o_4d, lse_4d * LOG2E))

    def _direct_4d_bwd(res_4d, do_4d):
      q_4d, k_4d, v_4d, o_4d, lse_bwd = res_4d
      dq_4d, dk_4d, dv_4d = champion.splash_attention_bwd_jax_k(
          q=q_4d,
          k=k_4d,
          v=v_4d,
          o=o_4d,
          do=do_4d,
          logsumexp=lse_bwd,
          segment_ids=SegmentIds(q=seg, kv=seg) if seg is not None else None,
          block_q=block_size,
          block_kv=block_size,
          is_causal=True,
          use_log2=True,
          dtype=q_4d.dtype,
      )
      return (dq_4d, dk_4d, dv_4d)

    _direct_4d_attn.defvjp(_direct_4d_fwd, _direct_4d_bwd)

    def _make_scanned_grad_fn(attn_fn):

      def _loss_fn(x_in, wq_in, wk_in, wv_in, wo_in):

        def _sub(h, wq_l, wk_l, wv_l, wo_l):
          q = jnp.einsum('bsd,dhk->bhsk', h, wq_l) * jnp.asarray(
              attn_scale, dtype=dtype
          )
          k = jnp.einsum('bsd,dhk->bhsk', h, wk_l)
          v = jnp.einsum('bsd,dhk->bhsk', h, wv_l)
          attn_out = attn_fn(q, k, v)
          return h + jnp.einsum('bhsk,hkd->bsd', attn_out, wo_l)

        def _body(carry, lw):
          wq_p, wk_p, wv_p, wo_p = lw
          carry = _sub(carry, wq_p[0], wk_p[0], wv_p[0], wo_p[0])
          carry = _sub(carry, wq_p[1], wk_p[1], wv_p[1], wo_p[1])
          return (carry, None)

        final_h, _ = jax.lax.scan(_body, x_in, (wq_in, wk_in, wv_in, wo_in))
        return jnp.sum(final_h.astype(jnp.float32) ** 2) / hidden_dim

      return jax.jit(jax.grad(_loss_fn, argnums=(0, 1, 2, 3, 4)))

    def _rel_fro(actual: jax.Array, ref: jax.Array) -> float:
      diff = np.asarray(actual, dtype=np.float32) - np.asarray(
          ref, dtype=np.float32
      )
      return float(
          np.linalg.norm(diff)
          / np.maximum(np.linalg.norm(np.asarray(ref, dtype=np.float32)), 1e-12)
      )

    grads_direct = jax.block_until_ready(
        _make_scanned_grad_fn(_direct_4d_attn)(x0, wq, wk, wv, wo)
    )
    grads_ref = jax.block_until_ready(
        _make_scanned_grad_fn(_ref_4d_attn)(x0, wq, wk, wv, wo)
    )
    for g_dir, g_ref in zip(grads_direct, grads_ref):
      self._assert_allclose(g_dir, g_ref, atol=0.01, rtol=0.01)
      self.assertLess(_rel_fro(g_dir, g_ref), 0.01)

  def test_backward_flops_counts_all_five_contractions(self):
    """Verifies SplashBwdConfig.backward_flops includes all 5 GEMM contractions."""
    cfg = configs.CANONICAL_BENCHMARK_CONFIGS[0]
    active_tokens = (
        cfg.batch_size * cfg.num_q_heads * cfg.q_seq_len * cfg.kv_seq_len * 0.5
    )
    expected_flops = active_tokens * (6.0 * cfg.dim_qk + 4.0 * cfg.dim_v)
    self.assertAlmostEqual(cfg.backward_flops, expected_flops)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
