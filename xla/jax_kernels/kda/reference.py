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

"""Pure JAX reference implementations for kda."""

import math
from typing import Any
import jax
import jax.numpy as jnp

RCP_LN2 = 1.0 / math.log(2)
_TPU_VMEM_USAGE_FRACTION = 0.9


def cdiv(x: Any, y: int) -> Any:
  return (x + y - 1) // y


def align_up(x: Any, align: int) -> Any:
  return cdiv(x, align) * align


def exp2(x: jax.Array) -> jax.Array:
  return jnp.exp2(x.astype(jnp.float32))


def l2norm_fwd(x: jax.Array, eps: float = 1e-06) -> tuple[jax.Array, jax.Array]:
  """Normalizes x along the last dimension and returns (normalized, rstd)."""
  x_f = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + eps)
  return ((x_f * rstd[..., None]).astype(x.dtype), rstd.astype(jnp.float32))


def segment_ids_to_seqlens(
    segment_ids: jax.Array, max_segs: int, chunk_size: int = 1
) -> jax.Array:
  """Converts 1 indexed segment IDs with 0 padding to cu_seqlens."""
  if segment_ids.ndim == 2:
    rows = [
        segment_ids_to_seqlens(segment_ids[b], max_segs, chunk_size)
        for b in range(segment_ids.shape[0])
    ]
    return jnp.stack(rows, axis=0)
  seg = segment_ids.reshape(-1)
  valid = seg != 0
  is_boundary = (
      jnp.concatenate([jnp.ones(1, dtype=jnp.bool_), seg[1:] != seg[:-1]])
      & valid
  )
  seg_idx = jnp.where(valid, jnp.cumsum(is_boundary.astype(jnp.int32)), 0)
  n_segs = seg_idx.max()
  n_real = jnp.sum(valid).astype(jnp.int32)
  is_end = (
      jnp.concatenate([seg[1:] != seg[:-1], jnp.ones(1, dtype=jnp.bool_)])
      & valid
  )
  prefix_len = jnp.cumsum(valid.astype(jnp.int32))
  drop_idx = jnp.asarray(max_segs + 1, dtype=jnp.int32)
  scatter_idx = jnp.where(is_end, seg_idx, drop_idx)
  scatter_val = jnp.where(is_end, prefix_len, 0)
  cu_seqlens = jnp.zeros((max_segs + 1,), dtype=jnp.int32)
  cu_seqlens = cu_seqlens.at[scatter_idx].max(scatter_val, mode='drop')
  out_idx = jnp.arange(max_segs + 1, dtype=jnp.int32)
  return jnp.where(out_idx > n_segs, n_real, cu_seqlens)


def segment_ids_to_cu_seqlens(
    segment_ids: jax.Array | None,
    *,
    initial_state: jax.Array | None,
    max_num_segments: int | None
) -> tuple[jax.Array | None, int | None]:
  """Derives cu_seqlens and max_num_segments from segment_ids."""
  if segment_ids is None:
    return (None, max_num_segments)
  if max_num_segments is None:
    if initial_state is None:
      raise ValueError(
          'max_num_segments is required when segment_ids is provided without'
          ' initial_state.'
      )
    max_num_segments = initial_state.shape[1]
  return (
      segment_ids_to_seqlens(segment_ids, max_segs=max_num_segments),
      max_num_segments,
  )


def align_seqs(
    tensors_4d: list[jax.Array],
    tensors_3d: list[jax.Array],
    cu_seqlens: jax.Array,
    align: int,
    aligned_cu_seqlens: jax.Array | None = None,
) -> tuple[list[jax.Array], list[jax.Array], jax.Array, jax.Array]:
  """Pads each variable length sequence to a multiple of align."""
  if cu_seqlens.ndim == 2:
    b_sz = cu_seqlens.shape[0]
    per_batch_4d: list[list[jax.Array]] = [[] for _ in tensors_4d]
    per_batch_3d: list[list[jax.Array]] = [[] for _ in tensors_3d]
    padded_cus = []
    t_aligned_sizes = []
    for b in range(b_sz):
      t4 = [t[:, b : b + 1, :, :] for t in tensors_4d]
      t3 = [t[:, b : b + 1, :] for t in tensors_3d]
      aligned_cu_b = (
          None if aligned_cu_seqlens is None else aligned_cu_seqlens[b]
      )
      aligned_4d, aligned_3d, padded_cu_b, _ = align_seqs(
          t4, t3, cu_seqlens[b], align, aligned_cu_seqlens=aligned_cu_b
      )
      for idx, a in enumerate(aligned_4d):
        per_batch_4d[idx].append(a)
      for idx, a in enumerate(aligned_3d):
        per_batch_3d[idx].append(a)
      padded_cus.append(padded_cu_b)
      t_aligned_sizes.append(aligned_4d[0].shape[2])
    t_max = max(t_aligned_sizes)

    def _pad_and_cat_4d(tensors_per_batch: list[jax.Array]) -> jax.Array:
      padded = []
      for t in tensors_per_batch:
        pad_len = t_max - t.shape[2]
        if pad_len > 0:
          t = jnp.pad(t, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        padded.append(t)
      return jnp.concatenate(padded, axis=1)

    def _pad_and_cat_3d(tensors_per_batch: list[jax.Array]) -> jax.Array:
      padded = []
      for t in tensors_per_batch:
        pad_len = t_max - t.shape[2]
        if pad_len > 0:
          t = jnp.pad(t, ((0, 0), (0, 0), (0, pad_len)))
        padded.append(t)
      return jnp.concatenate(padded, axis=1)

    out_4d = [_pad_and_cat_4d(per_batch_4d[i]) for i in range(len(tensors_4d))]
    out_3d = [_pad_and_cat_3d(per_batch_3d[i]) for i in range(len(tensors_3d))]
    stacked_cu = jnp.stack(padded_cus, axis=0)
    return (out_4d, out_3d, stacked_cu, cu_seqlens)
  n_segs = cu_seqlens.shape[0] - 1
  t_old = tensors_4d[0].shape[2]
  seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
  if aligned_cu_seqlens is None:
    padded_lens = (seg_lens + align - 1) // align * align
    padded_cu = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(padded_lens)]
    )
  else:
    padded_cu = aligned_cu_seqlens
  t_new = (t_old + n_segs * (align - 1) + align - 1) // align * align

  def _build_gather(i: jax.Array, gather_idx: jax.Array) -> jax.Array:
    old_start = cu_seqlens[i]
    new_start = padded_cu[i]
    sl = seg_lens[i]
    j = jnp.arange(t_new)
    in_seg = (j >= new_start) & (j < new_start + sl)
    src = old_start + (j - new_start)
    return jnp.where(in_seg, src, gather_idx)

  gather_idx = jnp.full(t_new, t_old, dtype=jnp.int32)
  gather_idx = jax.lax.fori_loop(0, n_segs, _build_gather, gather_idx)

  def _repack_4d(t: jax.Array) -> jax.Array:
    return jnp.pad(t, ((0, 0), (0, 0), (0, t_new - t_old), (0, 0)))[
        :, :, gather_idx
    ]

  def _repack_3d(t: jax.Array) -> jax.Array:
    return jnp.pad(t, ((0, 0), (0, 0), (0, t_new - t_old)))[:, :, gather_idx]

  return (
      [_repack_4d(t) for t in tensors_4d],
      [_repack_3d(t) for t in tensors_3d],
      padded_cu,
      cu_seqlens,
  )


def unalign_output(
    o: jax.Array,
    orig_cu_seqlens: jax.Array,
    aligned_cu_seqlens: jax.Array,
    t_out: int,
) -> jax.Array:
  """Scatters aligned output back to original variable length positions."""
  if orig_cu_seqlens.ndim == 2:
    b_sz = orig_cu_seqlens.shape[0]
    per_batch = []
    for b in range(b_sz):
      ob_slice = jax.lax.dynamic_slice_in_dim(o, b, 1, axis=1)
      ob = unalign_output(
          ob_slice, orig_cu_seqlens[b], aligned_cu_seqlens[b], t_out
      )
      per_batch.append(ob)
    return jnp.concatenate(per_batch, axis=1)
  n_segs = orig_cu_seqlens.shape[0] - 1
  orig_seg_lens = orig_cu_seqlens[1:] - orig_cu_seqlens[:-1]

  def _build_gather(i: jax.Array, gather_idx: jax.Array) -> jax.Array:
    orig_start = orig_cu_seqlens[i]
    aligned_start = aligned_cu_seqlens[i]
    sl = orig_seg_lens[i]
    j = jnp.arange(t_out)
    in_seg = (j >= orig_start) & (j < orig_start + sl)
    src = aligned_start + (j - orig_start)
    return jnp.where(in_seg, src, gather_idx)

  safe_default = aligned_cu_seqlens[-1]
  gather_idx = jnp.full(t_out, safe_default, dtype=jnp.int32)
  gather_idx = jax.lax.fori_loop(0, n_segs, _build_gather, gather_idx)
  return o[:, :, gather_idx]


def prepare_chunk_indices(
    cu_seqlens: jax.Array, chunk_size: int, max_t: int | jax.Array | None = None
) -> jax.Array:
  """Computes per chunk (seq_id, block_id) mapping from cu_seqlens."""
  if cu_seqlens.ndim == 2:
    rows = [
        prepare_chunk_indices(cu_seqlens[b], chunk_size, max_t=max_t)
        for b in range(cu_seqlens.shape[0])
    ]
    return jnp.stack(rows, axis=0)
  lens = cu_seqlens[1:] - cu_seqlens[:-1]
  n_chunks = cdiv(lens, chunk_size)
  num_seqs = len(lens)
  target_max_t = cu_seqlens[-1] if max_t is None else max_t
  total_nt = int(target_max_t) // chunk_size
  seq_ids = jnp.repeat(
      jnp.arange(num_seqs, dtype=jnp.int32),
      n_chunks,
      total_repeat_length=total_nt,
  )
  prefix_chunks = jnp.concatenate(
      [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(n_chunks)]
  )
  seq_offsets = jnp.repeat(
      prefix_chunks[:-1], n_chunks, total_repeat_length=total_nt
  )
  block_ids = jnp.arange(total_nt, dtype=jnp.int32) - seq_offsets
  return jnp.stack([seq_ids, block_ids], axis=1)


def kda_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    a_log: jax.Array | None = None,
    delta_time_bias: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
    use_gate_in_kernel: bool = False,
    segment_ids: jax.Array | None = None,
    lower_bound: float | None = None,
    max_num_segments: int | None = None,
    chunk_size: int = 64
) -> tuple[jax.Array, jax.Array | None]:
  """Computes KDA forward with an explicit token by token JAX recurrence."""
  del chunk_size
  heads, batch, seq_len, key_dim = q.shape
  value_dim = v.shape[-1]
  acc_dtype = jnp.float32
  output_dtype = q.dtype
  scale_val = float(key_dim ** (-0.5)) if scale is None else float(scale)
  if use_gate_in_kernel:
    if a_log is None:
      raise ValueError('a_log must be provided when use_gate_in_kernel=True.')
    g_f = g.astype(jnp.float32)
    if delta_time_bias is not None:
      g_f = g_f + delta_time_bias.astype(jnp.float32).reshape(
          heads, 1, 1, key_dim
      )
    a_val = jnp.exp(a_log.astype(jnp.float32)).reshape(heads, 1, 1, 1)
    if lower_bound is None:
      g = -a_val * jax.nn.softplus(g_f)
    else:
      g = lower_bound * jax.nn.sigmoid(a_val * g_f)
  if use_qk_l2norm:
    q_h, _ = l2norm_fwd(q.astype(acc_dtype))
    k_h, _ = l2norm_fwd(k.astype(acc_dtype))
  else:
    q_h = q.astype(acc_dtype)
    k_h = k.astype(acc_dtype)
  q_h = q_h * scale_val
  v_h = v.astype(acc_dtype)
  g_h = g.astype(acc_dtype)
  beta_h = beta.astype(acc_dtype)
  if initial_state is not None:
    num_states = initial_state.shape[1] if initial_state.ndim == 5 else 1
  elif segment_ids is None:
    num_states = 1
  elif max_num_segments is not None:
    num_states = max_num_segments
  else:
    raise ValueError(
        'max_num_segments is required when segment_ids is provided without'
        ' initial_state.'
    )
  states = jnp.zeros(
      (batch, num_states, heads, key_dim, value_dim), dtype=acc_dtype
  )
  if initial_state is not None:
    if initial_state.ndim == 4:
      states = states.at[:, 0].set(initial_state.astype(acc_dtype))
    else:
      states = states + initial_state.astype(acc_dtype)
  seg_ids_bt = (
      jnp.ones((batch, seq_len), dtype=jnp.int32)
      if segment_ids is None
      else segment_ids.astype(jnp.int32)
  )
  states_hb = jnp.transpose(states, (2, 0, 1, 3, 4))

  def _scan_one_hb(
      q_t_arr: jax.Array,
      k_t_arr: jax.Array,
      v_t_arr: jax.Array,
      g_t_arr: jax.Array,
      beta_t_arr: jax.Array,
      seg_t_arr: jax.Array,
      init_states_nkv: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:

    def _step(
        cur_states_nkv: jax.Array,
        xs: tuple[
            jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ],
    ) -> tuple[jax.Array, jax.Array]:
      q_t, k_t, v_t, g_t, beta_t, seg_id = xs
      state_idx = jnp.clip(seg_id - 1, 0, num_states - 1)
      valid = (seg_id > 0) & (seg_id <= num_states)
      previous_state = cur_states_nkv[state_idx]
      state = previous_state * jnp.exp(g_t)[:, None]
      prediction = jnp.matmul(
          k_t,
          state,
          precision=jax.lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )
      residual = v_t - prediction
      new_state = state + (beta_t * k_t)[:, None] * residual[None, :]
      out_t = jnp.matmul(
          q_t,
          new_state,
          precision=jax.lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )
      out_t = jnp.where(valid, out_t, jnp.zeros_like(out_t))
      updated_state = jnp.where(valid, new_state, previous_state)
      next_states_nkv = cur_states_nkv.at[state_idx].set(updated_state)
      return (next_states_nkv, out_t)

    return jax.lax.scan(
        _step,
        init_states_nkv,
        (q_t_arr, k_t_arr, v_t_arr, g_t_arr, beta_t_arr, seg_t_arr),
    )

  scan_b = jax.vmap(_scan_one_hb, in_axes=(0, 0, 0, 0, 0, 0, 0))
  scan_hb = jax.vmap(scan_b, in_axes=(0, 0, 0, 0, 0, None, 0))
  final_states_hb, output_h = scan_hb(
      q_h, k_h, v_h, g_h, beta_h, seg_ids_bt, states_hb
  )
  output = output_h.astype(output_dtype)
  final_state = (
      jnp.transpose(final_states_hb, (1, 2, 0, 3, 4))
      if output_final_state
      else None
  )
  return (output, final_state)


kda_ref = kda_reference


def _solve_unit_lower_triangular_batched(
    a: jax.Array, b: jax.Array
) -> jax.Array:
  """Solves (I + A) x = b for batched inputs via block forward substitution."""
  _, n, _ = b.shape
  bs = 16
  num_blocks = n // bs
  a = a.astype(jnp.float32)
  b = b.astype(jnp.float32)
  blocks = [b[:, i * bs : (i + 1) * bs, :] for i in range(num_blocks)]
  for i in range(num_blocks):
    start = i * bs
    end = (i + 1) * bs
    a_ii = a[:, start:end, start:end]
    x_block = blocks[i]
    rows = [x_block[:, r, :] for r in range(bs)]
    for j in range(bs):
      if j > 0:
        vec = a_ii[:, j, :j]
        mat = jnp.stack(rows[:j], axis=1)
        correction = jnp.matmul(vec[:, None, :], mat).squeeze(1)
        rows[j] = rows[j] - correction
    x_block = jnp.stack(rows, axis=1)
    blocks[i] = x_block
    if i < num_blocks - 1:
      x_rest = jnp.concatenate(blocks[i + 1 :], axis=1)
      a_rest = a[:, (i + 1) * bs :, start:end]
      update = jnp.matmul(a_rest, x_block, preferred_element_type=jnp.float32)
      x_rest = x_rest - update
      remaining = num_blocks - 1 - i
      for idx in range(remaining):
        blocks[i + 1 + idx] = x_rest[:, idx * bs : (idx + 1) * bs, :]
  return jnp.concatenate(blocks, axis=1)


def _solve_unit_lower_triangular_neumann_batched(
    a: jax.Array, b: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Solves (I + A) x = b with a blocked finite Neumann series."""
  n = a.shape[-1]
  block_size = 8
  num_blocks = n // block_size
  inv_dtype = jnp.float32
  identity = jnp.eye(n, dtype=inv_dtype)

  def _dot_batch(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        lhs,
        rhs,
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )

  a = a.astype(inv_dtype)
  block_ids = jnp.arange(n, dtype=jnp.int32) // block_size
  same_block = (block_ids[:, None] == block_ids[None, :]).astype(inv_dtype)
  a_diag = a * same_block[None]
  a_off_diag = a - a_diag
  neg_a_diag = -a_diag
  diag_inverse = identity[None] + neg_a_diag
  power = neg_a_diag
  num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[block_size]
  for _ in range(num_diag_steps):
    power = _dot_batch(power, power)
    diag_inverse = _dot_batch(diag_inverse, identity[None] + power)
  if num_blocks == 1:
    return (_dot_batch(diag_inverse, b.astype(inv_dtype)), diag_inverse)
  off_diag_and_rhs = jnp.concatenate([a_off_diag, b.astype(inv_dtype)], axis=-1)
  merged = _dot_batch(diag_inverse, off_diag_and_rhs)
  transformed_off_diag = merged[:, :, :n]
  transformed_rhs = merged[:, :, n:]
  neg_transformed_off_diag = -transformed_off_diag
  off_diag_inverse = identity[None] + neg_transformed_off_diag
  power = neg_transformed_off_diag
  log2_num_blocks = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[num_blocks]
  for _ in range(log2_num_blocks - 1):
    power = _dot_batch(power, power)
    off_diag_inverse = off_diag_inverse + _dot_batch(off_diag_inverse, power)
  result = _dot_batch(off_diag_inverse, transformed_rhs)
  inverse = _dot_batch(off_diag_inverse, diag_inverse)
  return (result, inverse)


def _fused_gate_intra_kernel(
    q_ref: Any,
    k_ref: Any,
    g_ref: Any,
    beta_ref: Any,
    v_ref: Any,
    a_log_ref: Any,
    delta_time_bias_ref: Any,
    u_out_ref: Any,
    w_out_ref: Any,
    qg_out_ref: Any,
    kg_out_ref: Any,
    aqk_out_ref: Any,
    akk_inv_out_ref: Any,
    g_cumsum_out_ref: Any,
    *,
    chunk_size: int,
    head_dim: int,
    value_dim: int,
    scale: float,
    cumsum_scale: float,
    fuse_cumsum: bool,
    disable_recompute: bool,
    safe_gate: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    mini_batch: int = 1
) -> None:
  """Fused Pallas kernel for gate activation, cumsum, Aqk/L, and solve."""
  dtype = q_ref.dtype
  bt = chunk_size
  bc = 16
  nc = bt // bc
  k_dim = head_dim
  v_dim = value_dim
  mb = mini_batch
  q = q_ref[:, 0, 0]
  k = k_ref[:, 0, 0]
  g = g_ref[:, 0, 0]
  beta = beta_ref[:, 0, 0]
  v = v_ref[:, 0, 0]
  g_f32 = g.astype(jnp.float32)
  if use_gate_in_kernel:
    dt_b = delta_time_bias_ref[:, 0, 0, 0]
    g_f32 = g_f32 + dt_b[:, None, :]
    a_val = a_log_ref[:, 0, 0, 0, 0]
    if lower_bound is None:
      g_f32 = -jnp.exp(a_val)[:, None, None] * jax.nn.softplus(g_f32)
    else:
      g_f32 = lower_bound * jax.nn.sigmoid(
          jnp.exp(a_val)[:, None, None] * g_f32
      )
  if fuse_cumsum:
    tril = jnp.tril(jnp.ones((bt, bt), dtype=jnp.float32))
    g_cumsum = (
        jax.lax.dot_general(
            tril,
            g_f32,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).transpose(1, 0, 2)
        * cumsum_scale
    )
  else:
    g_cumsum = g_f32
  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  beta_f32 = beta.astype(jnp.float32)
  ref_idx = bc // 2 if safe_gate else 0
  row_iota_bt_k = jax.lax.broadcasted_iota(jnp.int32, (bt, k_dim), dimension=0)
  row_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (bc, bt), dimension=0)
  col_iota_bc_bt = jax.lax.broadcasted_iota(jnp.int32, (bc, bt), dimension=1)
  aqk_rows = []
  l_rows = []
  for i_sc in range(nc):
    i_s = i_sc * bc
    q_i = q_f32[:, i_s : i_s + bc]
    k_i = k_f32[:, i_s : i_s + bc]
    g_i = g_cumsum[:, i_s : i_s + bc]
    beta_i = beta_f32[:, i_s : i_s + bc]
    gn_ref = g_i[:, ref_idx : ref_idx + 1, :]
    diff_i = g_i - gn_ref
    exp_diff_i = jnp.exp2(diff_i)
    q_eg = q_i * exp_diff_i
    k_eg = k_i * exp_diff_i
    valid_j = (row_iota_bt_k < i_s + bc).astype(jnp.float32)
    diff_j_safe = (gn_ref - g_cumsum) * valid_j[None]
    k_eng_full = k_f32 * jnp.exp2(diff_j_safe) * valid_j[None]
    qk_eg = jnp.concatenate([q_eg, k_eg], axis=1)
    qk_dot = jax.lax.dot_general(
        qk_eg,
        k_eng_full,
        (((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    aqk_row = qk_dot[:, :bc] * scale
    akk_row = qk_dot[:, bc:] * beta_i
    in_diag = (col_iota_bc_bt >= i_s) & (col_iota_bc_bt < i_s + bc)
    col_local = col_iota_bc_bt - i_s
    causal_diag = row_iota_bc_bt >= col_local
    strict_diag = row_iota_bc_bt > col_local
    aqk_keep = ~in_diag | causal_diag
    akk_keep = ~in_diag | strict_diag
    aqk_row = jnp.where(aqk_keep[None], aqk_row, jnp.float32(0.0))
    akk_row = jnp.where(akk_keep[None], akk_row, jnp.float32(0.0))
    aqk_rows.append(aqk_row)
    l_rows.append(akk_row)
  aqk = jnp.concatenate(aqk_rows, axis=1).astype(dtype)
  l_mat = jnp.concatenate(l_rows, axis=1)
  v_beta = v.astype(jnp.float32) * beta_f32
  k_eg_beta = k_f32 * jnp.exp2(g_cumsum) * beta_f32
  rhs = jnp.concatenate([v_beta, k_eg_beta], axis=-1)
  use_neumann = dtype != jnp.float32
  if use_neumann:
    result, a_inv = _solve_unit_lower_triangular_neumann_batched(l_mat, rhs)
  else:
    i_bt = jnp.eye(bt, dtype=jnp.float32)
    i_bt_batch = jnp.broadcast_to(i_bt, (mb, bt, bt))
    combined_b = jnp.concatenate([rhs, i_bt_batch], axis=-1)
    full_result = _solve_unit_lower_triangular_batched(l_mat, combined_b)
    result = full_result[:, :, : v_dim + k_dim]
    a_inv = full_result[:, :, v_dim + k_dim :]
  u = result[:, :, :v_dim]
  w = result[:, :, v_dim : v_dim + k_dim]
  g_last = g_cumsum[:, bt - 1 : bt, :]
  kg = k_f32 * exp2(g_last - g_cumsum)
  qg = q_f32 * exp2(g_cumsum) if disable_recompute else jnp.zeros_like(q_f32)
  u_out_ref[:, 0, 0] = u.astype(u_out_ref.dtype)
  w_out_ref[:, 0, 0] = w.astype(w_out_ref.dtype)
  qg_out_ref[:, 0, 0] = qg.astype(qg_out_ref.dtype)
  kg_out_ref[:, 0, 0] = kg.astype(kg_out_ref.dtype)
  aqk_out_ref[:, 0, 0] = aqk.astype(aqk_out_ref.dtype)
  akk_inv_out_ref[:, 0, 0] = a_inv.astype(akk_inv_out_ref.dtype)
  g_cumsum_out_ref[:, 0, 0] = g_cumsum


def reference_kda_fwd_intra_fused(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float,
    chunk_size: int = 64,
    safe_gate: bool = True,
    disable_recompute: bool = False,
    cumsum_scale: float = RCP_LN2,
    a_log: jax.Array | None = None,
    delta_time_bias: jax.Array | None = None,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
  """Pure JAX reference for fused KDA intra chunk solve."""
  heads, batch, seq_len, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = chunk_size
  nc = seq_len // bt
  q_f = q.reshape(heads, batch, nc, bt, k_dim).astype(jnp.float32)
  k_f = k.reshape(heads, batch, nc, bt, k_dim).astype(jnp.float32)
  v_f = v.reshape(heads, batch, nc, bt, v_dim).astype(jnp.float32)
  g_f = g.reshape(heads, batch, nc, bt, k_dim).astype(jnp.float32)
  beta_f = beta.reshape(heads, batch, nc, bt, 1).astype(jnp.float32)
  if use_gate_in_kernel:
    assert a_log is not None
    if delta_time_bias is not None:
      g_f = g_f + delta_time_bias.astype(jnp.float32).reshape(
          heads, 1, 1, 1, k_dim
      )
    a_val = jnp.exp(a_log.astype(jnp.float32)).reshape(heads, 1, 1, 1, 1)
    g_f = (
        -a_val * jax.nn.softplus(g_f)
        if lower_bound is None
        else lower_bound * jax.nn.sigmoid(a_val * g_f)
    )
  g_cumsum = jnp.cumsum(g_f, axis=3) * cumsum_scale
  bc = 16
  ref_idx = bc // 2 if safe_gate else 0
  row_iota_bt_k = jax.lax.broadcasted_iota(jnp.int32, (bt, k_dim), dimension=0)
  row_iota = jax.lax.broadcasted_iota(jnp.int32, (bc, bt), dimension=0)
  col_iota = jax.lax.broadcasted_iota(jnp.int32, (bc, bt), dimension=1)
  aqk_rows, l_rows = ([], [])
  for i_sc in range(bt // bc):
    i_s = i_sc * bc
    q_i = q_f[:, :, :, i_s : i_s + bc, :]
    k_i = k_f[:, :, :, i_s : i_s + bc, :]
    g_i = g_cumsum[:, :, :, i_s : i_s + bc, :]
    beta_i = beta_f[:, :, :, i_s : i_s + bc, :]
    gn_ref = g_i[:, :, :, ref_idx : ref_idx + 1, :]
    exp_i = jnp.exp2(g_i - gn_ref)
    valid_j = (row_iota_bt_k < i_s + bc).astype(jnp.float32)
    k_eng = (
        k_f
        * jnp.exp2((gn_ref - g_cumsum) * valid_j[None, None, None])
        * valid_j[None, None, None]
    )
    aqk_r = (
        jnp.einsum(
            'hbcsd,hbcjd->hbcsj',
            q_i * exp_i,
            k_eng,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    akk_r = (
        jnp.einsum(
            'hbcsd,hbcjd->hbcsj',
            k_i * exp_i,
            k_eng,
            precision=jax.lax.Precision.HIGHEST,
        )
        * beta_i
    )
    in_diag = (col_iota >= i_s) & (col_iota < i_s + bc)
    col_loc = col_iota - i_s
    aqk_r = jnp.where(~in_diag | (row_iota >= col_loc), aqk_r, 0.0)
    akk_r = jnp.where(~in_diag | (row_iota > col_loc), akk_r, 0.0)
    aqk_rows.append(aqk_r)
    l_rows.append(akk_r)
  aqk = jnp.concatenate(aqk_rows, axis=3).astype(q.dtype)
  l_mat = jnp.concatenate(l_rows, axis=3)
  eye = jnp.eye(bt, dtype=jnp.float32)
  a_inv = jax.scipy.linalg.solve_triangular(
      jnp.broadcast_to(eye, l_mat.shape) + l_mat,
      jnp.broadcast_to(eye, l_mat.shape),
      lower=True,
      unit_diagonal=True,
  )
  rhs = jnp.concatenate(
      [v_f * beta_f, k_f * jnp.exp2(g_cumsum) * beta_f], axis=-1
  )
  sol = jnp.einsum(
      'hbcij,hbcjd->hbcid', a_inv, rhs, precision=jax.lax.Precision.HIGHEST
  )
  u_out = sol[..., :v_dim].reshape(heads, batch, seq_len, v_dim).astype(v.dtype)
  w_out = sol[..., v_dim:].reshape(heads, batch, seq_len, k_dim).astype(k.dtype)
  g_last = g_cumsum[:, :, :, bt - 1 : bt, :]
  kg_out = (
      (k_f * jnp.exp2(g_last - g_cumsum))
      .reshape(heads, batch, seq_len, k_dim)
      .astype(k.dtype)
  )
  qg_out = (
      (q_f * jnp.exp2(g_cumsum) if disable_recompute else jnp.zeros_like(q_f))
      .reshape(heads, batch, seq_len, k_dim)
      .astype(q.dtype)
  )
  return (
      w_out,
      u_out,
      qg_out,
      kg_out,
      aqk.reshape(heads, batch, seq_len, bt),
      a_inv.reshape(heads, batch, seq_len, bt).astype(q.dtype),
      g_cumsum.reshape(heads, batch, seq_len, k_dim),
  )


def reference_chunk_kda_fwd_h_o_varlen(
    w: jax.Array,
    u: jax.Array,
    kg: jax.Array,
    gk: jax.Array,
    q: jax.Array,
    a: jax.Array,
    cu_seqlens: jax.Array,
    chunk_indices: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    scale: float = 1.0,
    chunk_size: int = 64,
    store_h: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, None]:
  """Pure JAX reference for Stage 3+4 KDA chunk state propagation and output."""
  heads, batch, seq_len, k_dim = q.shape
  v_dim = u.shape[-1]
  bt = chunk_size
  nc = seq_len // bt
  if cu_seqlens.ndim == 1:
    cu_seqlens = jnp.broadcast_to(
        cu_seqlens[None, :], (batch, cu_seqlens.shape[0])
    )
  n_segs = cu_seqlens.shape[-1] - 1
  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, bt, max_t=seq_len)
  if chunk_indices.ndim == 2:
    chunk_indices = jnp.broadcast_to(chunk_indices[None, :, :], (batch, nc, 2))
  o_batches, ht_batches, h_batches = ([], [], [])
  for b_idx in range(batch):
    h_segs = (
        jnp.zeros((n_segs, heads, k_dim, v_dim), dtype=jnp.float32)
        if initial_state is None
        else initial_state[b_idx].astype(jnp.float32)
    )
    o_chunks, h_chunks = ([], [])
    for c_idx in range(nc):
      seq_id = int(chunk_indices[b_idx, c_idx, 0])
      blk_id = int(chunk_indices[b_idx, c_idx, 1])
      seg_len = int(cu_seqlens[b_idx, seq_id + 1] - cu_seqlens[b_idx, seq_id])
      if blk_id != 0:
        h_cur = h_segs[seq_id]
      elif initial_state is None:
        h_cur = jnp.zeros((heads, k_dim, v_dim), dtype=jnp.float32)
      else:
        h_cur = initial_state[b_idx, seq_id].astype(jnp.float32)
      h_chunks.append(h_cur)
      sl = slice(c_idx * bt, (c_idx + 1) * bt)
      w_c = w[:, b_idx, sl, :].astype(jnp.float32)
      u_c = u[:, b_idx, sl, :].astype(jnp.float32)
      kg_c = kg[:, b_idx, sl, :].astype(jnp.float32)
      gk_c = gk[:, b_idx, sl, :].astype(jnp.float32)
      q_c = q[:, b_idx, sl, :].astype(jnp.float32)
      a_c = a[:, b_idx, sl, :].astype(jnp.float32)
      v_new = u_c - jnp.einsum(
          'htk,hkv->htv', w_c, h_cur, precision=jax.lax.Precision.HIGHEST
      )
      qg_c = q_c * jnp.exp2(gk_c) * scale
      o_c = jnp.einsum(
          'htk,hkv->htv', qg_c, h_cur, precision=jax.lax.Precision.HIGHEST
      ) + jnp.einsum(
          'hts,hsv->htv', a_c, v_new, precision=jax.lax.Precision.HIGHEST
      )
      o_chunks.append(o_c.astype(q.dtype))
      if seg_len > 0 and blk_id * bt < seg_len:
        valid_rows = min(bt, seg_len - blk_id * bt)
        gk_last = gk_c[:, valid_rows - 1, :]
        v_mask = (jnp.arange(bt) < valid_rows)[None, :, None]
        v_new_m = jnp.where(v_mask, v_new, 0.0)
        kg_m = jnp.where(v_mask, kg_c, 0.0)
        h_next = h_cur * jnp.exp2(gk_last)[:, :, None] + jnp.einsum(
            'htk,htv->hkv', kg_m, v_new_m, precision=jax.lax.Precision.HIGHEST
        )
        h_segs = h_segs.at[seq_id].set(h_next)
    o_batches.append(jnp.concatenate(o_chunks, axis=1))
    ht_batches.append(h_segs)
    h_batches.append(jnp.stack(h_chunks, axis=1))
  o_all = jnp.stack(o_batches, axis=1)
  ht_all = jnp.stack(ht_batches, axis=0) if output_final_state else None
  h_all = None
  if store_h:
    h_all = jnp.transpose(jnp.stack(h_batches, axis=0), (1, 0, 2, 3, 4)).astype(
        q.dtype
    )
  return (o_all, ht_all, h_all, None)


def reference_chunk_gated_delta_rule_fwd_h_pre_process(
    k: jax.Array,
    w: jax.Array,
    u: jax.Array,
    gk: jax.Array,
    cu_seqlens: jax.Array,
    chunk_indices: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
  """Pure JAX reference for KDA context parallel `(S_ext, M)` pre process."""
  heads, batch, t_local, k_dim = k.shape
  v_dim = u.shape[-1]
  bt = chunk_size
  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, bt, max_t=t_local)
  s_batches, m_batches = ([], [])
  eye_k = jnp.eye(k_dim, dtype=jnp.float32)
  for b_idx in range(batch):
    cu_b = cu_seqlens[b_idx] if cu_seqlens.ndim == 2 else cu_seqlens
    ci_b = chunk_indices[b_idx] if chunk_indices.ndim == 3 else chunk_indices
    seg_lens = jnp.diff(cu_b)
    valid_segs = [i for i in range(seg_lens.shape[0]) if int(seg_lens[i]) > 0]
    last_seg = valid_segs[-1] if valid_segs else 0
    bos = int(cu_b[last_seg])
    eos = int(cu_b[last_seg + 1])
    h_acc = jnp.zeros((heads, k_dim, v_dim), dtype=jnp.float32)
    m_acc = jnp.broadcast_to(eye_k, (heads, k_dim, k_dim))
    for c_idx in range(ci_b.shape[0]):
      seq_idx = int(ci_b[c_idx, 0])
      t0 = c_idx * bt
      if seq_idx != last_seg:
        continue
      if t0 == bos:
        h_acc = jnp.zeros((heads, k_dim, v_dim), dtype=jnp.float32)
        m_acc = jnp.broadcast_to(eye_k, (heads, k_dim, k_dim))
      sl = slice(t0, t0 + bt)
      k_c = k[:, b_idx, sl, :].astype(jnp.float32)
      w_c = w[:, b_idx, sl, :].astype(jnp.float32)
      u_c = u[:, b_idx, sl, :].astype(jnp.float32)
      gk_last = gk[:, b_idx, t0 + bt - 1, :].astype(jnp.float32)
      decay = jnp.exp2(jnp.maximum(gk_last, -126.0))
      diag = decay[..., None] * eye_k
      kw = jnp.einsum(
          'htk,htm->hkm', k_c, w_c, precision=jax.lax.Precision.HIGHEST
      )
      m_step = diag - kw
      ds = jnp.einsum(
          'htk,htv->hkv', k_c, u_c, precision=jax.lax.Precision.HIGHEST
      )
      h_acc = (
          jnp.einsum(
              'hkm,hmv->hkv', m_step, h_acc, precision=jax.lax.Precision.HIGHEST
          )
          + ds
      )
      m_acc = jnp.einsum(
          'hkm,hmn->hkn', m_step, m_acc, precision=jax.lax.Precision.HIGHEST
      )
      if t0 + bt >= eos:
        break
    s_batches.append(h_acc[:, None, :, :].astype(jnp.float32))
    m_batches.append(m_acc[:, None, :, :].astype(jnp.float32))
  return (
      jnp.concatenate(s_batches, axis=1),
      jnp.concatenate(m_batches, axis=1),
  )
