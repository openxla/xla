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

"""Pure JAX reference implementations for mamba2."""

import jax
import jax.numpy as jnp


def segsum(x: jax.Array) -> jax.Array:
  """Computes segmented sums for a 1D array along the last dimension."""
  seq_len = x.shape[-1]
  x_rep = jnp.repeat(x[..., None], seq_len, axis=-1)
  lower_strict = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool), k=-1)
  x_masked = jnp.where(lower_strict, x_rep, 0.0)
  x_segsum = jnp.cumsum(x_masked, axis=-2)
  lower_unit = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool), k=0)
  return jnp.where(lower_unit, x_segsum, -jnp.inf)


def ssd_reference(
    *,
    dt: jax.Array,
    x: jax.Array,
    b: jax.Array,
    c: jax.Array,
    log_a: jax.Array,
    d: jax.Array,
    dt_bias: jax.Array,
    block_len: int = 64
) -> jax.Array:
  """Reference Mamba2 chunked State Space Duality (SSD) in JAX."""
  orig_dtype = x.dtype
  batch_size, seq_len, d_inner = x.shape
  n_heads = dt.shape[-1]
  d_head = d_inner // n_heads
  d_state = b.shape[-1]
  x_f32 = x.astype(jnp.float32).reshape(batch_size, seq_len, n_heads, d_head)
  dt_f32 = jax.nn.softplus(dt.astype(jnp.float32) + dt_bias.astype(jnp.float32))
  x_times_dt = x_f32 * dt_f32[..., None]
  a_times_dt = -jnp.exp(log_a.astype(jnp.float32)) * dt_f32
  num_chunks = seq_len // block_len
  x_chunked = x_times_dt.reshape(
      batch_size, num_chunks, block_len, n_heads, d_head
  )
  a_chunked = a_times_dt.reshape(batch_size, num_chunks, block_len, n_heads)
  b_chunked = b.astype(jnp.float32).reshape(
      batch_size, num_chunks, block_len, d_state
  )
  c_chunked = c.astype(jnp.float32).reshape(
      batch_size, num_chunks, block_len, d_state
  )
  a_chunked = jnp.transpose(a_chunked, (0, 3, 1, 2))
  a_cumsum = jnp.cumsum(a_chunked, axis=-1)
  l_mat = jnp.exp(segsum(a_chunked))
  y_diag = jnp.einsum(
      'bcln,bcsn,bhcls,bcshp->bclhp',
      c_chunked,
      b_chunked,
      l_mat,
      x_chunked,
      precision=jax.lax.Precision.HIGHEST,
  )
  decay_states = jnp.exp(a_cumsum[..., -1:] - a_cumsum)
  states = jnp.einsum(
      'bcln,bhcl,bclhp->bchpn',
      b_chunked,
      decay_states,
      x_chunked,
      precision=jax.lax.Precision.HIGHEST,
  )
  initial_states = jnp.zeros_like(states[:, :1])
  states_cat = jnp.concatenate([initial_states, states], axis=1)
  a_chunk_sum = a_cumsum[..., -1]
  a_chunk_sum_padded = jnp.pad(a_chunk_sum, ((0, 0), (0, 0), (1, 0)))
  decay_chunk = jnp.exp(segsum(a_chunk_sum_padded))
  new_states = jnp.einsum(
      'bhzc,bchpn->bzhpn',
      decay_chunk,
      states_cat,
      precision=jax.lax.Precision.HIGHEST,
  )
  states_out = new_states[:, :-1]
  state_decay_out = jnp.exp(a_cumsum)
  y_off = jnp.einsum(
      'bcln,bchpn,bhcl->bclhp',
      c_chunked,
      states_out,
      state_decay_out,
      precision=jax.lax.Precision.HIGHEST,
  )
  y = y_diag + y_off
  x_orig_chunked = x_f32.reshape(
      batch_size, num_chunks, block_len, n_heads, d_head
  )
  d_f32 = d.astype(jnp.float32)
  y = y + x_orig_chunked * d_f32[None, None, None, :, None]
  return y.reshape(batch_size, seq_len, d_inner).astype(orig_dtype)


ssd_ref = ssd_reference
