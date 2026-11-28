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

"""Unit and Hypothesis correctness tests for jax_k Mamba2 SSD kernels."""

from typing import Any
from absl.testing import absltest
import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
import numpy as np
from xla.jax_kernels import mamba2
from xla.jax_kernels.mamba2 import champion
from xla.jax_kernels.mamba2 import configs
from xla.jax_kernels.mamba2 import reference as baseline


class Mamba2KernelsTest(absltest.TestCase):

  @hypothesis.given(
      seq_len=st.sampled_from([512, 1024]),
      block_len=st.sampled_from([64, 128]),
      dtype=st.sampled_from([jnp.float32, jnp.bfloat16]),
  )
  @hypothesis.settings(max_examples=6, deadline=None)
  def test_0_ssd_jax_k_vs_tokamax_pallas_and_ref(
      self, seq_len: int, block_len: int, dtype: Any
  ):
    batch_size = 2
    d_inner = 512
    n_heads = 8
    d_state = 64
    key = jax.random.PRNGKey(100 + seq_len + block_len)
    inputs = configs.make_ssd_inputs(
        key,
        batch_size=batch_size,
        seq_len=seq_len,
        d_inner=d_inner,
        n_heads=n_heads,
        d_state=d_state,
        dtype=dtype,
    )
    actual_jax_k = jax.jit(
        champion.ssd_jax_k, static_argnames=('block_len', 'n_block_size')
    )(**inputs, block_len=block_len, n_block_size=64).block_until_ready()
    expected_ref = jax.jit(baseline.ssd_ref, static_argnames=('block_len',))(
        **inputs, block_len=block_len
    ).block_until_ready()
    tol = 0.01 if dtype == jnp.bfloat16 else 0.0001
    np.testing.assert_allclose(actual_jax_k, expected_ref, rtol=tol, atol=tol)

  def test_1_ssd_pipelined_chunk_blocks_vs_tokamax(self):
    batch_size = 2
    seq_len = 512
    block_len = 64
    d_inner = 512
    n_heads = 8
    d_state = 64
    key = jax.random.PRNGKey(202)
    inputs = configs.make_ssd_inputs(
        key,
        batch_size=batch_size,
        seq_len=seq_len,
        d_inner=d_inner,
        n_heads=n_heads,
        d_state=d_state,
        dtype=jnp.float32,
    )
    actual_jax_k_pipelined = jax.jit(
        mamba2.ssd_jax_k, static_argnames=('block_len', 'n_block_size')
    )(**inputs, block_len=block_len, n_block_size=2).block_until_ready()
    expected_ref = jax.jit(mamba2.ssd_ref, static_argnames=('block_len',))(
        **inputs, block_len=block_len
    ).block_until_ready()
    np.testing.assert_allclose(
        actual_jax_k_pipelined, expected_ref, rtol=0.0001, atol=0.0001
    )

  def test_2_ssd_grid_direct_vs_pipelined(self):
    batch_size, n_heads, num_chunks, block_len, d_head, d_state = (
        2,
        8,
        8,
        64,
        64,
        64,
    )
    key = jax.random.PRNGKey(303)
    kx, ka, kb, kc = jax.random.split(key, 4)
    x_dt = (
        jax.random.normal(
            kx,
            (batch_size, n_heads, num_chunks, block_len, d_head),
            dtype=jnp.float32,
        )
        * 0.1
    )
    a_dt = -jax.random.uniform(
        ka,
        (batch_size, n_heads, num_chunks, block_len),
        minval=0.001,
        maxval=0.05,
        dtype=jnp.float32,
    )
    b = (
        jax.random.normal(
            kb, (batch_size, num_chunks, block_len, d_state), dtype=jnp.float32
        )
        * 0.2
    )
    c = (
        jax.random.normal(
            kc, (batch_size, num_chunks, block_len, d_state), dtype=jnp.float32
        )
        * 0.2
    )
    out = jnp.zeros_like(x_dt)
    actual_direct = jax.jit(
        champion.ssd_grid, static_argnames=('n_block_size',)
    )(x_dt, a_dt, b, c, out, n_block_size=64).block_until_ready()
    actual_pipelined = jax.jit(
        champion.ssd_grid, static_argnames=('n_block_size',)
    )(x_dt, a_dt, b, c, out, n_block_size=2).block_until_ready()
    np.testing.assert_allclose(
        actual_direct, actual_pipelined, rtol=0.0001, atol=0.0001
    )

  def test_3_hlo_invariants(self):
    batch_size, n_heads, num_chunks, block_len, d_head, d_state = (
        1,
        8,
        8,
        64,
        64,
        64,
    )
    x_dt = jnp.ones(
        (batch_size, n_heads, num_chunks, block_len, d_head), dtype=jnp.float32
    )
    a_dt = jnp.full(
        (batch_size, n_heads, num_chunks, block_len), -0.01, dtype=jnp.float32
    )
    b = jnp.ones(
        (batch_size, num_chunks, block_len, d_state), dtype=jnp.float32
    )
    c = jnp.ones(
        (batch_size, num_chunks, block_len, d_state), dtype=jnp.float32
    )
    out = jnp.zeros_like(x_dt)
    jit_fn = jax.jit(champion.ssd_grid, static_argnames=('n_block_size',))
    lowered = jit_fn.lower(x_dt, a_dt, b, c, out, n_block_size=2)
    hlo_pre = lowered.as_text()
    self.assertIn('xla_disable_while_loop_copies', hlo_pre)
    self.assertIn('xla_disjoint_read_write_regions', hlo_pre)
    self.assertIn('stablehlo.while', hlo_pre)
    compiled = lowered.compile()
    hlo_opt = compiled.as_text()
    self.assertIsNotNone(hlo_opt)
    self.assertIn('unrolled_2x', hlo_opt)


if __name__ == '__main__':
  absltest.main()
