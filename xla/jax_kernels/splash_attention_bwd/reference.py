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

"""Pure JAX reference and mask helpers for Splash Attention backward."""

from collections.abc import Sequence
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xla.jax_kernels.splash_attention_bwd import configs

LN2 = configs.LN2
LOG2E = configs.LOG2E


class Mask:
  """Base 2D or 3D boolean attention mask."""

  _arr: np.ndarray

  def __init__(self, arr: np.ndarray):
    self._arr = np.asarray(arr, dtype=np.bool_)

  @property
  def shape(self) -> tuple[int, ...]:
    return self._arr.shape

  def __getitem__(self, idx: Any) -> np.ndarray:
    return self._arr[idx]

  def __or__(self, other: "Mask") -> "Mask":
    return Mask(self._arr | other._arr)

  def __and__(self, other: "Mask") -> "Mask":
    return Mask(self._arr & other._arr)


class NumpyMask(Mask):
  """Wraps a boolean NumPy array as a Splash mask."""

  def __init__(self, array: np.ndarray):
    super().__init__(np.asarray(array, dtype=np.bool_))


class FullMask(Mask):
  """Unmasked 2D full attention mask."""

  def __init__(self, shape: tuple[int, int]):
    super().__init__(np.ones(shape, dtype=np.bool_))


class CausalMask(Mask):
  """Lower triangular causal 2D attention mask."""

  def __init__(self, shape: tuple[int, int], offset: int = 0):
    q_len, kv_len = shape
    q_idx = np.arange(q_len, dtype=np.int32)[:, None]
    kv_idx = np.arange(kv_len, dtype=np.int32)[None, :]
    super().__init__(q_idx + offset >= kv_idx)


class LocalMask(Mask):
  """Sliding window local 2D attention mask."""

  def __init__(
      self,
      shape: tuple[int, int],
      window_size: tuple[int | None, int | None],
      offset: int = 0,
  ):
    q_len, kv_len = shape
    q_idx = np.arange(q_len, dtype=np.int32)[:, None]
    kv_idx = np.arange(kv_len, dtype=np.int32)[None, :]
    arr = np.ones(shape, dtype=np.bool_)
    left, right = window_size
    if left is not None:
      arr = arr & (q_idx - left + offset <= kv_idx)
    if right is not None:
      arr = arr & (q_idx + right + offset >= kv_idx)
    super().__init__(arr)


class ChunkedCausalMask(Mask):
  """Block diagonal chunked causal 2D attention mask."""

  def __init__(self, shape: tuple[int, int], chunk_size: int):
    if chunk_size <= 0:
      raise ValueError("chunk_size must be positive")
    q_len, kv_len = shape
    q_idx = np.arange(q_len, dtype=np.int32)[:, None]
    kv_idx = np.arange(kv_len, dtype=np.int32)[None, :]
    same = (q_idx // chunk_size) == (kv_idx // chunk_size)
    super().__init__(same & (q_idx >= kv_idx))


class MultiHeadMask(Mask):
  """Stacks per head 2D masks into a 3D `[H, Q, KV]` mask."""

  def __init__(self, masks: Sequence[Mask]):
    self.masks = tuple(masks)
    super().__init__(np.stack([m[:, :] for m in self.masks], axis=0))


def _normalize_mask_3d(mask: Any, num_q_heads: int) -> jax.Array:
  """Normalizes a 2D or 3D mask into a boolean `[H, Q, KV]` JAX array."""
  if isinstance(mask, (jax.Array, np.ndarray)):
    arr = jnp.asarray(mask, dtype=jnp.bool_)
  else:
    arr = jnp.asarray(
        mask[:, :] if len(mask.shape) == 2 else mask[:, :, :], dtype=jnp.bool_
    )
  if arr.ndim == 2:
    arr = jnp.broadcast_to(arr[None, :, :], (num_q_heads, *arr.shape))
  elif arr.shape[0] == 1 and num_q_heads > 1:
    arr = jnp.broadcast_to(arr, (num_q_heads, *arr.shape[1:]))
  return arr


def _single_head_fwd(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: Any,
    sinks: Any,
    mask_value: float,
    attn_logits_soft_cap: float | None,
) -> tuple[jax.Array, jax.Array]:
  """Computes single head Splash forward output and natural logsumexp."""
  logits = jnp.einsum("sd,td->st", q.astype(jnp.float32), k.astype(jnp.float32))
  if segment_ids is not None:
    mask = jnp.logical_and(
        mask, segment_ids.q[:, None] == segment_ids.kv[None, :]
    )
  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(logits / attn_logits_soft_cap) * attn_logits_soft_cap
  logits = jnp.where(mask, logits, mask_value)
  m = logits.max(axis=-1)
  if sinks is not None:
    sinks = sinks.astype(logits.dtype)
    m = jnp.maximum(m, sinks)
  s = jnp.exp(logits - m[..., None])
  l = s.sum(axis=-1) + (0.0 if sinks is None else jnp.exp(sinks - m))
  p = s / l[..., None]
  o = jnp.einsum("st,td->sd", p, v.astype(jnp.float32))
  lse = m + jnp.log(l)
  return o, lse


def reference_splash_attention_fwd(
    mask: Any,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: Any = None,
    sinks: Any = None,
    is_mqa: bool = False,
    attn_logits_soft_cap: float | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Computes pure JAX reference Splash forward output and logsumexp."""
  num_q_heads = q.shape[-3]
  mask_arr = _normalize_mask_3d(mask, num_q_heads)
  sinks_h_ax = 0 if sinks is not None else None
  fn = functools.partial(
      _single_head_fwd,
      mask_value=configs.DEFAULT_MASK_VALUE,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )
  if is_mqa:
    head_fn = jax.vmap(fn, in_axes=(0, 0, None, None, None, sinks_h_ax))
  else:
    if k.shape[1] < num_q_heads:
      rep = num_q_heads // k.shape[1]
      k = jnp.repeat(k, rep, axis=1)
      v = jnp.repeat(v, rep, axis=1)
    head_fn = jax.vmap(fn, in_axes=(0, 0, 0, 0, None, sinks_h_ax))
  sinks_b_ax = 0 if (sinks is not None and jnp.ndim(sinks) == 2) else None
  mask_b_ax = 0 if mask_arr.ndim == 4 else None
  if mask_arr.ndim == 4 and mask_arr.shape[0] == 1 and q.shape[0] > 1:
    mask_arr = jnp.broadcast_to(mask_arr, (q.shape[0], *mask_arr.shape[1:]))
  batch_fn = jax.vmap(
      head_fn,
      in_axes=(
          mask_b_ax,
          0,
          0,
          0,
          0 if segment_ids is not None else None,
          sinks_b_ax,
      ),
  )
  return batch_fn(mask_arr, q, k, v, segment_ids, sinks)


def _single_head_bwd(
    mask_s: jax.Array,
    q_s: jax.Array,
    k_s: jax.Array,
    v_s: jax.Array,
    seg_s: Any,
    sinks_s: Any,
    o_s: jax.Array,
    lse_s: jax.Array,
    do_s: jax.Array,
    attn_logits_soft_cap: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
  """Computes single head Splash backward gradients `(dq, dk, dv, dsinks)`."""
  uncapped = jnp.einsum(
      "qc,kc->qk", q_s, k_s, preferred_element_type=jnp.float32
  )
  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(uncapped / attn_logits_soft_cap) * attn_logits_soft_cap
  else:
    logits = uncapped
  if seg_s is not None:
    mask_s = jnp.logical_and(mask_s, seg_s.q[:, None] == seg_s.kv[None, :])
  logits = jnp.where(mask_s, logits, configs.DEFAULT_MASK_VALUE)
  p = jnp.exp(logits - lse_s[..., None])
  do_f = do_s.astype(jnp.float32)
  dv = jnp.einsum("pt,pd->td", p, do_f).astype(v_s.dtype)
  dp = jnp.einsum("pd,td->pt", do_f, v_s.astype(jnp.float32))
  di = jnp.sum(o_s.astype(jnp.float32) * do_f, axis=-1)[..., None]
  ds = (dp - di) * p
  if attn_logits_soft_cap is not None:
    d = jnp.tanh(uncapped / attn_logits_soft_cap)
    g = ds * (1.0 - d)
    ds = g + g * d
  dk = jnp.einsum("sd,st->td", q_s.astype(jnp.float32), ds).astype(k_s.dtype)
  dq = jnp.einsum("st,td->sd", ds, k_s.astype(jnp.float32)).astype(q_s.dtype)
  dsinks = None
  if sinks_s is not None:
    sinks_exp = -jnp.exp(
        sinks_s[..., None, None].astype(jnp.float32)
        - lse_s[..., None].astype(jnp.float32)
    )
    dsinks = jnp.sum(sinks_exp * o_s.astype(jnp.float32) * do_f).astype(
        sinks_s.dtype
    )
  return dq, dk, dv, dsinks


def reference_splash_attention_bwd(
    mask: Any,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: Any,
    sinks: Any,
    o_ref: jax.Array,
    logsumexp: jax.Array,
    do: jax.Array,
    num_q_heads: int,
    num_kv_heads: int,
    attn_logits_soft_cap: float | None = None,
    is_mqa: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
  """Computes pure JAX reference Splash backward gradients."""
  mask_arr = _normalize_mask_3d(mask, num_q_heads)
  is_grouped = not is_mqa and num_kv_heads < num_q_heads
  head_multiplier = num_q_heads // num_kv_heads
  sinks_h_ax = 0 if sinks is not None else None
  fn = functools.partial(
      _single_head_bwd, attn_logits_soft_cap=attn_logits_soft_cap
  )
  if is_mqa:
    inner = jax.vmap(fn, in_axes=(0, 0, None, None, None, sinks_h_ax, 0, 0, 0))
  else:
    inner = jax.vmap(fn, in_axes=(0, 0, 0, 0, None, sinks_h_ax, 0, 0, 0))
    if is_grouped:
      k = jnp.repeat(k, head_multiplier, axis=1)
      v = jnp.repeat(v, head_multiplier, axis=1)
  sinks_b_ax = 0 if (sinks is not None and jnp.ndim(sinks) == 2) else None
  mask_b_ax = 0 if mask_arr.ndim == 4 else None
  if mask_arr.ndim == 4 and mask_arr.shape[0] == 1 and q.shape[0] > 1:
    mask_arr = jnp.broadcast_to(mask_arr, (q.shape[0], *mask_arr.shape[1:]))
  batch_fn = jax.vmap(
      inner,
      in_axes=(
          mask_b_ax,
          0,
          0,
          0,
          0 if segment_ids is not None else None,
          sinks_b_ax,
          0,
          0,
          0,
      ),
  )
  dq_ref, dk_ref, dv_ref, dsinks_ref = batch_fn(
      mask_arr,
      q,
      k,
      v,
      segment_ids,
      sinks,
      o_ref.astype(jnp.float32),
      logsumexp,
      do.astype(jnp.float32),
  )
  if is_mqa:
    dk_ref, dv_ref = dk_ref.sum(axis=1), dv_ref.sum(axis=1)
  elif is_grouped:
    dk_ref = dk_ref.reshape(
        dk_ref.shape[0], num_kv_heads, head_multiplier, *dk_ref.shape[2:]
    ).sum(axis=2)
    dv_ref = dv_ref.reshape(
        dv_ref.shape[0], num_kv_heads, head_multiplier, *dv_ref.shape[2:]
    ).sum(axis=2)
  if dsinks_ref is not None and jnp.ndim(sinks) == 1:
    dsinks_ref = dsinks_ref.sum(axis=0)
  return dq_ref, dk_ref, dv_ref, dsinks_ref
