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

"""Unit and Hypothesis correctness tests for jax_k KDA forward kernels."""

from typing import Any
from absl.testing import absltest
import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
import numpy as np
from xla.jax_kernels import kda
from xla.jax_kernels.kda import champion
from xla.jax_kernels.kda import configs
from xla.jax_kernels.kda import reference as baseline


class KdaKernelsTest(absltest.TestCase):

  @hypothesis.given(
      seq_len=st.sampled_from([64, 256, 512]),
      heads=st.sampled_from([2, 8]),
      dtype=st.sampled_from([jnp.bfloat16, jnp.float32]),
      use_gate_in_kernel=st.sampled_from([False, True]),
  )
  @hypothesis.settings(max_examples=6, deadline=None)
  def test_0_intra_fused_jax_k_vs_pallas(
      self, seq_len: int, heads: int, dtype: Any, use_gate_in_kernel: bool
  ):
    cfg = configs.KdaBenchmarkConfig(
        name='intra_test',
        batch_size=1,
        seq_len=seq_len,
        heads=heads,
        key_dim=128,
        value_dim=128,
        dtype=dtype,
        use_qk_l2norm=False,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=True,
        lower_bound=-5.0 if use_gate_in_kernel else None,
    )
    key = jax.random.PRNGKey(100 + seq_len + heads)
    inp = configs.make_kda_inputs(key, cfg)
    jax_k_res = champion.jax_k_kda_fwd_intra_fused(
        q=inp['q'],
        k=inp['k'],
        v=inp['v'],
        g=inp['g'],
        beta=inp['beta'],
        scale=cfg.scale,
        chunk_size=64,
        safe_gate=True,
        disable_recompute=True,
        a_log=inp['a_log'],
        delta_time_bias=inp['delta_time_bias'],
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=cfg.lower_bound,
    )
    pallas_res = baseline.reference_kda_fwd_intra_fused(
        q=inp['q'],
        k=inp['k'],
        v=inp['v'],
        g=inp['g'],
        beta=inp['beta'],
        scale=cfg.scale,
        chunk_size=64,
        safe_gate=True,
        disable_recompute=True,
        a_log=inp['a_log'],
        delta_time_bias=inp['delta_time_bias'],
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=cfg.lower_bound,
    )
    for idx, name in enumerate(
        ['w', 'u', 'qg', 'kg', 'Aqk', 'Akk', 'g_cumsum']
    ):
      n_val = jax_k_res[idx]
      p_val = pallas_res[idx]
      assert n_val is not None and p_val is not None
      np.testing.assert_allclose(
          n_val,
          p_val,
          rtol=0.01,
          atol=0.01,
          err_msg=f'Mismatch in Stage 1+2 output {name}',
      )

  def test_1_stage34_varlen_and_fixed_vs_pallas(self):
    for dtype in [jnp.float32, jnp.bfloat16]:
      cfg = configs.KdaBenchmarkConfig(
          name='stage34_test',
          batch_size=2,
          seq_len=256,
          heads=4,
          key_dim=128,
          value_dim=128,
          dtype=dtype,
          use_qk_l2norm=False,
          use_gate_in_kernel=True,
          safe_gate=True,
          lower_bound=-5.0,
          use_initial_state=True,
          output_final_state=True,
      )
      key = jax.random.PRNGKey(202)
      inp = configs.make_kda_inputs(key, cfg)
      w, u, _, kg, aqk, _, g_cumsum = champion.jax_k_kda_fwd_intra_fused(
          q=inp['q'],
          k=inp['k'],
          v=inp['v'],
          g=inp['g'],
          beta=inp['beta'],
          scale=cfg.scale,
          chunk_size=64,
          safe_gate=True,
          disable_recompute=False,
          a_log=inp['a_log'],
          delta_time_bias=inp['delta_time_bias'],
          use_gate_in_kernel=True,
          lower_bound=-5.0,
      )
      cu_seqlens = jnp.broadcast_to(
          jnp.asarray([0, 256], dtype=jnp.int32), (2, 2)
      )
      chunk_indices = champion.prepare_chunk_indices(cu_seqlens, 64, max_t=256)
      o_jax_k, ht_jax_k, h_jax_k, _ = champion.chunk_kda_fwd_h_o_varlen(
          w=w,
          u=u,
          kg=kg,
          gk=g_cumsum,
          q=inp['q'],
          a=aqk,
          cu_seqlens=cu_seqlens,
          chunk_indices=chunk_indices,
          initial_state=inp['initial_state'],
          output_final_state=True,
          scale=cfg.scale,
          chunk_size=64,
          store_h=True,
      )
      o_pallas, ht_pallas, h_pallas, _ = (
          baseline.reference_chunk_kda_fwd_h_o_varlen(
              w=w,
              u=u,
              kg=kg,
              gk=g_cumsum,
              q=inp['q'],
              a=aqk,
              cu_seqlens=cu_seqlens,
              chunk_indices=chunk_indices,
              initial_state=inp['initial_state'],
              output_final_state=True,
              scale=cfg.scale,
              chunk_size=64,
              store_h=True,
          )
      )
      assert (
          ht_jax_k is not None
          and ht_pallas is not None
          and (h_jax_k is not None)
          and (h_pallas is not None)
      )
      np.testing.assert_allclose(o_jax_k, o_pallas, rtol=0.01, atol=0.01)
      np.testing.assert_allclose(ht_jax_k, ht_pallas, rtol=0.01, atol=0.01)
      np.testing.assert_allclose(h_jax_k, h_pallas, rtol=0.01, atol=0.01)

  def test_2_cp_pre_process_jax_k_vs_pallas(self):
    heads, batch, t_local, k_dim, v_dim = (4, 2, 256, 128, 128)
    key = jax.random.PRNGKey(303)
    kk, kw, ku, kgk = jax.random.split(key, 4)
    k = (
        jax.random.normal(
            kk, (heads, batch, t_local, k_dim), dtype=jnp.bfloat16
        )
        * 0.1
    )
    w = (
        jax.random.normal(
            kw, (heads, batch, t_local, k_dim), dtype=jnp.bfloat16
        )
        * 0.1
    )
    u = (
        jax.random.normal(
            ku, (heads, batch, t_local, v_dim), dtype=jnp.bfloat16
        )
        * 0.1
    )
    gk = -jnp.abs(
        jax.random.normal(
            kgk, (heads, batch, t_local, k_dim), dtype=jnp.float32
        )
    )
    cu_seqlens = jnp.asarray(
        [[0, 64, 256, 256], [0, 128, 256, 256]], dtype=jnp.int32
    )
    chunk_indices = champion.prepare_chunk_indices(
        cu_seqlens, 64, max_t=t_local
    )
    s_jax_k, m_jax_k = champion.chunk_gated_delta_rule_fwd_h_pre_process(
        k=k,
        w=w,
        u=u,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=64,
    )
    s_pallas, m_pallas = (
        baseline.reference_chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            w=w,
            u=w,
            gk=gk,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=64,
        )
    )
    s_pallas_u, m_pallas_u = (
        baseline.reference_chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            w=w,
            u=u,
            gk=gk,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=64,
        )
    )
    del s_pallas, m_pallas
    np.testing.assert_allclose(s_jax_k, s_pallas_u, rtol=0.01, atol=0.01)
    np.testing.assert_allclose(m_jax_k, m_pallas_u, rtol=0.01, atol=0.01)

  def test_3_end_to_end_fixed_and_varlen_vs_pallas_and_reference(self):
    test_configs = [
        configs.KdaBenchmarkConfig(
            name='fixed_bf16_t1024',
            batch_size=1,
            seq_len=1024,
            heads=8,
            key_dim=128,
            value_dim=128,
            dtype=jnp.bfloat16,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_initial_state=True,
            output_final_state=True,
        ),
        configs.KdaBenchmarkConfig(
            name='varlen_bf16_t256',
            batch_size=1,
            seq_len=256,
            heads=4,
            key_dim=128,
            value_dim=128,
            dtype=jnp.bfloat16,
            use_qk_l2norm=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_initial_state=True,
            output_final_state=True,
            seq_lens=(45, 80, 20),
            max_num_segments=4,
        ),
        configs.KdaBenchmarkConfig(
            name='varlen_fp32_batch2',
            batch_size=2,
            seq_len=192,
            heads=4,
            key_dim=64,
            value_dim=64,
            dtype=jnp.float32,
            use_qk_l2norm=False,
            use_gate_in_kernel=False,
            safe_gate=True,
            lower_bound=None,
            use_initial_state=True,
            output_final_state=True,
            batch_seq_lens=((64, 128), (128, 64)),
            max_num_segments=2,
        ),
    ]
    for idx, cfg in enumerate(test_configs):
      key = jax.random.PRNGKey(400 + idx)
      inp = configs.make_kda_inputs(key, cfg)
      out_jax_k, state_jax_k = kda.kda_fwd_jax_k(**inp)
      out_ref, state_ref = baseline.kda_reference(
          inp['q'],
          inp['k'],
          inp['v'],
          inp['g'],
          inp['beta'],
          a_log=inp['a_log'],
          delta_time_bias=inp['delta_time_bias'],
          scale=inp['scale'],
          initial_state=inp['initial_state'],
          output_final_state=inp['output_final_state'],
          use_qk_l2norm=inp['use_qk_l2norm'],
          use_gate_in_kernel=inp['use_gate_in_kernel'],
          segment_ids=inp['segment_ids'],
          lower_bound=inp['lower_bound'],
          max_num_segments=inp['max_num_segments'],
      )
      np.testing.assert_allclose(
          out_jax_k,
          out_ref,
          rtol=0.01,
          atol=0.01,
          err_msg=f'Output mismatch vs reference in {cfg.name}',
      )
      if cfg.output_final_state:
        assert state_jax_k is not None and state_ref is not None
        if cfg.dtype == jnp.float32:
          np.testing.assert_allclose(
              state_jax_k,
              state_ref,
              rtol=0.01,
              atol=0.01,
              err_msg=f'Final state mismatch vs reference in {cfg.name}',
          )

  def test_4_hlo_invariants(self):
    heads, batch, seq_len, k_dim, v_dim, bt = (4, 1, 2048, 128, 128, 64)
    w = jnp.zeros((heads, batch, seq_len, k_dim), dtype=jnp.bfloat16)
    u = jnp.zeros((heads, batch, seq_len, v_dim), dtype=jnp.bfloat16)
    kg = jnp.zeros((heads, batch, seq_len, k_dim), dtype=jnp.bfloat16)
    gk = jnp.zeros((heads, batch, seq_len, k_dim), dtype=jnp.float32)
    q = jnp.zeros((heads, batch, seq_len, k_dim), dtype=jnp.bfloat16)
    a = jnp.zeros((heads, batch, seq_len, bt), dtype=jnp.bfloat16)
    jit_fn = jax.jit(
        champion.kda_fwd_h_o_grid,
        static_argnames=('scale', 'chunk_size', 'c_block_size'),
    )
    lowered = jit_fn.lower(
        w,
        u,
        kg,
        gk,
        q,
        a,
        None,
        None,
        scale=0.125,
        chunk_size=64,
        c_block_size=4,
    )
    hlo_pre = lowered.as_text()
    self.assertIn('xla_disable_while_loop_copies', hlo_pre)
    self.assertIn('xla_disjoint_read_write_regions', hlo_pre)
    self.assertIn('stablehlo.while', hlo_pre)
    compiled = lowered.compile()
    hlo_opt = compiled.as_text()
    self.assertIsNotNone(hlo_opt)
    self.assertTrue('while' in hlo_opt or 'unrolled_2x' in hlo_opt)


if __name__ == '__main__':
  absltest.main()
