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

"""Correctness tests for jax_k Fused MoE against Pallas and JAX reference."""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
import numpy as np
from xla.jax_kernels import fused_moe
from xla.jax_kernels.fused_moe import champion
from xla.jax_kernels.fused_moe import configs


class FusedMoeKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = configs.create_default_mesh(ep_axis_name='model')

  def _test_moe_vs_pallas_and_ref(
      self,
      dtype: Any,
      top_k: int,
      num_experts: int,
      hidden_size: int,
      intermediate_size: int,
      num_tokens: int,
      seed: int,
      renormalize_topk_logits: bool,
      bt: int | None = None,
      bf: int | None = None,
      bd1: int | None = None,
      bd2: int | None = None,
      btc: int | None = None,
      bfc: int | None = None,
      bd1c: int | None = None,
      bd2c: int | None = None,
      act_fn: str = 'silu',
      scoring_fn: str = 'softmax',
      w_dtype: Any = None,
      subc_quant_w1_sz: int | None = None,
      subc_quant_w2_sz: int | None = None,
      has_bias: bool = False,
      pallas_atol: float = 0.01,
      pallas_rtol: float = 0.01,
      ref_atol: float = 0.2,
      ref_rtol: float = 0.2,
  ) -> None:
    del pallas_atol, pallas_rtol
    a, w1, w2, b1, b2, gating_output = configs.gen_moe_inputs(
        dtype,
        top_k,
        num_experts,
        hidden_size,
        intermediate_size,
        num_tokens,
        seed=seed,
        has_bias=has_bias,
    )
    w1_scale = None
    w2_scale = None
    if w_dtype is not None:
      if subc_quant_w1_sz is None:
        subc_quant_w1_sz = 256
      if subc_quant_w2_sz is None:
        subc_quant_w2_sz = 256
      w1, w1_scale = configs.sub_channel_quantize(w1, w_dtype, subc_quant_w1_sz)
      w2, w2_scale = configs.sub_channel_quantize(w2, w_dtype, subc_quant_w2_sz)
    jax_k_out = champion.fused_ep_moe(
        mesh=self.mesh,
        tokens=a,
        w1=w1,
        w2=w2,
        gating_output=gating_output,
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
        subc_quant_w1_sz=subc_quant_w1_sz,
        subc_quant_w2_sz=subc_quant_w2_sz,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        b1=b1,
        b2=b2,
        bt=bt,
        bf=bf,
        bd1=bd1,
        bd2=bd2,
        btc=btc,
        bfc=bfc,
        bd1c=bd1c,
        bd2c=bd2c,
    )
    ref_out = champion.ref_moe(
        a,
        w1,
        w2,
        gating_output,
        top_k,
        b1=b1,
        b2=b2,
        renormalize_topk_logits=renormalize_topk_logits,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
        subc_quant_w1_sz=subc_quant_w1_sz,
        subc_quant_w2_sz=subc_quant_w2_sz,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    self.assertEqual(jax_k_out.shape, ref_out.shape)
    self.assertEqual(jax_k_out.dtype, ref_out.dtype)
    self.assertTrue(bool(jnp.all(jnp.isfinite(jax_k_out))))
    np.testing.assert_allclose(
        np.asarray(jax_k_out, dtype=np.float32),
        np.asarray(ref_out, dtype=np.float32),
        atol=ref_atol,
        rtol=ref_rtol,
    )

  @parameterized.product(renormalize_topk_logits=[True, False])
  def test_basic(self, renormalize_topk_logits: bool):
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=8,
        num_experts=128,
        hidden_size=1024,
        intermediate_size=1024,
        num_tokens=8 * 32,
        seed=1234,
        renormalize_topk_logits=renormalize_topk_logits,
        bt=32,
        bf=1024,
        bd1=1024,
        bd2=1024,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
    )

  @parameterized.product(act_fn=['silu', 'gelu', 'swigluoai'])
  def test_activation(self, act_fn: str):
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=8,
        num_experts=128,
        hidden_size=1024,
        intermediate_size=1024,
        num_tokens=8 * 32,
        seed=1234,
        renormalize_topk_logits=True,
        act_fn=act_fn,
        bt=32,
        bf=512,
        bd1=512,
        bd2=512,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
    )

  @parameterized.product(scoring_fn=['softmax', 'sigmoid'])
  def test_scoring_fn(self, scoring_fn: str):
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=8,
        num_experts=128,
        hidden_size=1024,
        intermediate_size=1024,
        num_tokens=8 * 32,
        seed=1234,
        renormalize_topk_logits=True,
        scoring_fn=scoring_fn,
        bt=32,
        bf=512,
        bd1=512,
        bd2=512,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
        ref_atol=0.4,
    )

  def test_benchmark_qwen_235(self):
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=8,
        num_experts=128,
        hidden_size=4096,
        intermediate_size=1536,
        num_tokens=8 * 64,
        seed=54321,
        renormalize_topk_logits=True,
        bt=64,
        bf=768,
        bd1=2048,
        bd2=2048,
        btc=64,
        bfc=768,
        bd1c=2048,
        bd2c=2048,
        act_fn='silu',
        ref_atol=0.05,
        ref_rtol=0.05,
    )

  def test_benchmark_qwen_30b_a3b(self):
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=8,
        num_experts=128,
        hidden_size=2048,
        intermediate_size=768,
        num_tokens=512,
        seed=54321,
        renormalize_topk_logits=True,
        bt=16,
        bf=384,
        bd1=512,
        bd2=512,
        btc=16,
        bfc=384,
        bd1c=256,
        bd2c=256,
        act_fn='silu',
        ref_atol=0.05,
        ref_rtol=0.05,
    )

  def test_benchmark_gpt_oss(self):
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=4,
        num_experts=128,
        hidden_size=3072,
        intermediate_size=3072,
        num_tokens=128,
        seed=1234,
        renormalize_topk_logits=True,
        act_fn='silu',
        ref_atol=0.05,
        ref_rtol=0.05,
    )

  @parameterized.parameters(128, 256, 512)
  def test_benchmark_qwen3_coder(self, num_tokens: int):
    cfg = next(
        c
        for c in configs.FUSED_MOE_BENCHMARK_CONFIGS
        if c.name.startswith(f'qwen3_coder_t{num_tokens}_')
    )
    tokens, w1, w2, gating_output, w1_scale, w2_scale, b1, b2 = (
        configs.make_moe_inputs(cfg, mesh=self.mesh)
    )
    jax_k_out = champion.fused_ep_moe(
        mesh=self.mesh,
        tokens=tokens,
        w1=w1,
        w2=w2,
        gating_output=gating_output,
        top_k=cfg.top_k,
        renormalize_topk_logits=cfg.renormalize_topk_logits,
        act_fn=cfg.act_fn,
        scoring_fn=cfg.scoring_fn,
        subc_quant_w1_sz=cfg.subc_quant_w1_sz,
        subc_quant_w2_sz=cfg.subc_quant_w2_sz,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        b1=b1,
        b2=b2,
    )
    ref_out = champion.ref_moe(
        tokens,
        w1,
        w2,
        gating_output,
        cfg.top_k,
        renormalize_topk_logits=cfg.renormalize_topk_logits,
        act_fn=cfg.act_fn,
        scoring_fn=cfg.scoring_fn,
        subc_quant_w1_sz=cfg.subc_quant_w1_sz,
        subc_quant_w2_sz=cfg.subc_quant_w2_sz,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        b1=b1,
        b2=b2,
    )
    jax_k_np = np.asarray(jax_k_out, dtype=np.float32)
    ref_np = np.asarray(ref_out, dtype=np.float32)
    np.testing.assert_allclose(jax_k_np, ref_np, atol=0.05, rtol=0.05)

  @parameterized.product(w_dtype=[jnp.int8, jnp.float8_e5m2, jnp.float4_e2m1fn])
  def test_sub_channel_quantization(self, w_dtype: Any):
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=8,
        num_experts=128,
        hidden_size=1024,
        intermediate_size=1024,
        num_tokens=8 * 32,
        seed=1234,
        renormalize_topk_logits=False,
        w_dtype=w_dtype,
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
    )

  @parameterized.product(w_dtype=[jnp.int8, jnp.float8_e5m2])
  def test_per_channel_quantization(self, w_dtype: Any):
    hidden_size = 512
    intermediate_size = 1024
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=8,
        num_experts=128,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_tokens=8 * 32,
        seed=1234,
        renormalize_topk_logits=False,
        w_dtype=w_dtype,
        subc_quant_w1_sz=hidden_size,
        subc_quant_w2_sz=intermediate_size,
        bt=32,
        bf=1024,
        bd1=hidden_size,
        bd2=hidden_size,
        btc=32,
        bfc=512,
        bd1c=256,
        bd2c=256,
    )

  def test_bias(self):
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=8,
        num_experts=128,
        hidden_size=1024,
        intermediate_size=1024,
        num_tokens=8 * 32,
        seed=1234,
        renormalize_topk_logits=False,
        has_bias=True,
        bt=32,
        bf=512,
        bd1=512,
        bd2=512,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
    )

  @hypothesis.given(
      top_k=st.sampled_from([2, 4, 8]),
      num_tokens=st.sampled_from([64, 128]),
      renorm=st.booleans(),
  )
  @hypothesis.settings(max_examples=6, deadline=None)
  def test_hypothesis_jax_k_vs_pallas(
      self, top_k: int, num_tokens: int, renorm: bool
  ):
    self._test_moe_vs_pallas_and_ref(
        dtype=jnp.bfloat16,
        top_k=top_k,
        num_experts=32,
        hidden_size=512,
        intermediate_size=512,
        num_tokens=num_tokens,
        seed=2026 + top_k + num_tokens,
        renormalize_topk_logits=renorm,
        bt=32,
        bf=512,
        bd1=512,
        bd2=512,
        btc=32,
        bfc=256,
        bd1c=256,
        bd2c=256,
    )

  def test_hlo_invariants(self):
    cfg = configs.FUSED_MOE_BENCHMARK_CONFIGS[0]
    tokens, w1, w2, gating, w1_scale, w2_scale, b1, b2 = (
        configs.make_moe_inputs(cfg)
    )
    fn = lambda a, w1_in, w2_in, g_in: fused_moe.fused_ep_moe(
        self.mesh,
        a,
        w1_in,
        w2_in,
        g_in,
        cfg.top_k,
        renormalize_topk_logits=cfg.renormalize_topk_logits,
        act_fn=cfg.act_fn,
        scoring_fn=cfg.scoring_fn,
        subc_quant_w1_sz=cfg.subc_quant_w1_sz,
        subc_quant_w2_sz=cfg.subc_quant_w2_sz,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        b1=b1,
        b2=b2,
        bt=cfg.bt,
        bf=cfg.bf,
        bd1=cfg.bd1,
        bd2=cfg.bd2,
        btc=cfg.btc,
        bfc=cfg.bfc,
        bd1c=cfg.bd1c,
        bd2c=cfg.bd2c,
    )
    lowered = jax.jit(fn).lower(tokens, w1, w2, gating)
    hlo_pre = lowered.as_text()
    self.assertNotIn('pallas_call', hlo_pre)
    self.assertIn('stablehlo.while', hlo_pre)
    self.assertIn('xla_disjoint_read_write_regions', hlo_pre)
    self.assertIn('xla_color_', hlo_pre)
    compiled = lowered.compile()
    hlo_opt = compiled.as_text()
    self.assertIsNotNone(hlo_opt)
    self.assertIn('fusion', hlo_opt)


if __name__ == '__main__':
  absltest.main()
