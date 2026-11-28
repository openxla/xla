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

"""Kimi Delta Attention (KDA) forward kernels in jax_k."""

import dataclasses
import functools
import inspect
import math
from typing import Any

import jax
from jax.experimental import xla_metadata
import jax.numpy as jnp

# pylint: disable=invalid-name,missing-class-docstring,missing-function-docstring,broad-exception-caught,g-complex-comprehension,g-long-ternary
_JAX_K_FUSION_UID = 0


def _jax_k_next_fuid(off: int = 0) -> str:
  global _JAX_K_FUSION_UID
  _JAX_K_FUSION_UID += 1
  return str(off + _JAX_K_FUSION_UID)


def _jax_k_block_slice(
    src: jax.Array,
    idx: tuple[Any, ...],
    block_shape: Any,
    *,
    immediate_revisiting: bool = True,
) -> jax.Array:
  b_sz = tuple(int(s) for s in block_shape)
  if globals().get("_JAX_K_PAD_DISC", False):
    globals()["_jax_k_rec_pad"](src, b_sz, idx, False)
    return jnp.zeros(b_sz, dtype=src.dtype)
  if any(d < s for d, s in zip(src.shape, b_sz)):
    src = jax.lax.pad(
        src,
        jnp.zeros((), dtype=src.dtype),
        tuple((0, max(0, s - d), 0) for d, s in zip(src.shape, b_sz)),
    )
  if immediate_revisiting:
    return jax.lax.dynamic_slice(src, idx, b_sz)
  with xla_metadata.set_xla_metadata(
      xla_while_loop_async_immediate_revisiting="false"
  ):
    return jax.lax.dynamic_slice(src, idx, b_sz)


def _jax_k_block_update(
    dst: jax.Array,
    tile: jax.Array,
    start_indices: tuple[Any, ...],
    *,
    bound: Any = None,
    in_pipelined_loop: bool = True,
    disjoint_read_write_regions: bool = True,
) -> jax.Array:
  if globals().get("_JAX_K_PAD_DISC", False):
    globals()["_jax_k_rec_pad"](dst, tile.shape, start_indices, True)
    return dst
  idx = tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices)
  orig_s: tuple[int, ...] | None = None
  if any(d < s for d, s in zip(dst.shape, tile.shape)):
    orig_s = dst.shape
    dst = jax.lax.pad(
        dst,
        jnp.zeros((), dtype=dst.dtype),
        tuple((0, max(0, s - d), 0) for d, s in zip(orig_s, tile.shape)),
    )
  if bound is not None:
    b0, b1 = (
        (bound[0], bound[1])
        if isinstance(bound, (tuple, list)) and len(bound) == 2
        else (0, bound)
    )
    rows = idx[0] + jnp.arange(tile.shape[0], dtype=jnp.int32).reshape(
        (tile.shape[0],) + (1,) * (tile.ndim - 1)
    )
    tile = jnp.where(
        (rows >= jnp.asarray(b0, dtype=jnp.int32))
        & (rows < jnp.asarray(b1, dtype=jnp.int32)),
        tile,
        jax.lax.dynamic_slice(dst, idx, tile.shape),
    )
  meta = {} if in_pipelined_loop else {"ColorInputToHBM": "false"}
  if disjoint_read_write_regions:
    meta["xla_disjoint_read_write_regions"] = "true"
  with xla_metadata.set_xla_metadata(**meta):
    res = jax.lax.dynamic_update_slice(dst, tile, idx)
  return res[tuple(slice(0, d) for d in orig_s)] if orig_s is not None else res


_JAX_K_LOOP_DEPTH = 0
_JAX_K_PAD_DISC = False
_JAX_K_CARRY_REQS: dict[int, Any] = {}
_JAX_K_CLOS_REQS: dict[int, Any] = {}
_JAX_K_CARRY_IDS: dict[int, Any] = {}
_JAX_K_SRC_MAP: dict[int, tuple[Any, Any]] = {}


def _jax_k_set_cidx(v: Any, idx: int) -> Any:
  if v is not None:
    _JAX_K_CARRY_IDS[id(v)] = (v, idx)
    try:
      object.__setattr__(v, "_jax_k_cidx", idx)
    except Exception:
      pass
  return v


def _jax_k_get_cidx(v: Any) -> int | None:
  if v is None:
    return None
  q, seen = [(v, 0)], set()
  while q:
    c, d = q.pop(0)
    if id(c) in seen or d > 6:
      continue
    seen.add(id(c))
    cd = getattr(c, "_jax_k_cidx", None)
    if isinstance(cd, int):
      return cd
    ce = _JAX_K_CARRY_IDS.get(id(c))
    if ce is not None and ce[0] is c:
      return ce[1]
    eq = getattr(c, "parent", None)
    if eq is not None:
      for t in getattr(eq, "in_tracers", ()):
        if getattr(t, "ndim", None) == getattr(v, "ndim", None):
          q.append((t, d + 1))
  return None


def _jax_k_get_src(v: Any) -> Any:
  se = _JAX_K_SRC_MAP.get(id(v))
  if se is not None and se[0] is v:
    return se[1]
  q, seen = [(v, 0)], set()
  while q:
    c, d = q.pop(0)
    if id(c) in seen or d > 6:
      continue
    seen.add(id(c))
    ce = _JAX_K_SRC_MAP.get(id(c))
    if ce is not None and ce[0] is c:
      return ce[1]
    eq = getattr(c, "parent", None)
    if eq is not None:
      ins = [
          t
          for t in getattr(eq, "in_tracers", ())
          if getattr(t, "ndim", None) == getattr(v, "ndim", None)
      ]
      if ins:
        q.extend((t, d + 1) for t in ins)
      else:
        return c
  return v


def _jax_k_is_ragged(iv: Any) -> bool:
  q, seen = [(iv, 0)], set()
  while q:
    c, d = q.pop(0)
    if id(c) in seen or d > 4:
      continue
    seen.add(id(c))
    eq = getattr(c, "parent", None)
    if eq is None:
      continue
    if getattr(eq.primitive, "name", "") in (
        "add",
        "sub",
        "min",
        "max",
        "select_n",
        "clamp",
        "gather",
        "dynamic_slice",
    ):
      return True
    q.extend((t, d + 1) for t in getattr(eq, "in_tracers", ()))
  return False


def _jax_k_rec_pad(
    t: Any, b_sz: tuple[int, ...], idx: Any, is_upd: bool
) -> None:
  if not _JAX_K_PAD_DISC or t is None or not hasattr(t, "shape"):
    return
  ci = _jax_k_get_cidx(t)
  if ci is not None and ci in _JAX_K_CARRY_REQS:
    orig_s, reqs = _JAX_K_CARRY_REQS[ci]
  else:
    rt = _jax_k_get_src(t)
    if not hasattr(rt, "shape"):
      return
    orig_s, reqs = _JAX_K_CLOS_REQS.setdefault(
        id(rt),
        (rt, tuple(int(d) for d in rt.shape), [int(d) for d in rt.shape]),
    )[1:]
  if len(b_sz) != len(orig_s) or len(idx) != len(orig_s):
    return
  for d, (do, bd, iv) in enumerate(zip(orig_s, b_sz, idx)):
    if isinstance(iv, int) and not isinstance(iv, bool) and iv == 0:
      rq = max(do, bd)
    elif bd == 1 and not is_upd:
      rq = max(1, do)
    else:
      al = ((max(1, do) + bd - 1) // bd) * bd
      rq = al + bd if _jax_k_is_ragged(iv) else al
    if isinstance(iv, int) and not isinstance(iv, bool) and iv + bd > rq:
      rq = iv + bd
    if rq > reqs[d]:
      reqs[d] = rq


def _jax_k_collect_cells(fn: Any) -> list[Any]:
  cells, seen = [], set()

  def _w(o: Any) -> None:
    if id(o) in seen:
      return
    seen.add(id(o))
    u = inspect.unwrap(o) if callable(o) else o
    if isinstance(u, functools.partial):
      _w(u.func)
      for a in u.args:
        _w(a)
      for v in (u.keywords or {}).values():
        _w(v)
    for c in getattr(u, "__closure__", None) or ():
      cells.append(c)
      try:
        _w(c.cell_contents)
      except ValueError:
        pass

  _w(fn)
  return cells


def _jax_k_rep_tree(
    o: Any, pmap: dict[int, Any], drest: list[Any], seen: set[int]
) -> tuple[Any, bool]:
  if id(o) in pmap:
    return pmap[id(o)], True
  if id(o) in seen:
    return o, False
  if isinstance(o, tuple):
    it, ch = (
        zip(*(_jax_k_rep_tree(x, pmap, drest, seen) for x in o))
        if o
        else ((), ())
    )
    return (tuple(it), True) if any(ch) else (o, False)
  if isinstance(o, list):
    seen.add(id(o))
    ch = False
    for i, x in enumerate(o):
      nx, xc = _jax_k_rep_tree(x, pmap, drest, seen)
      if xc:
        drest.append((o, i, x))
        o[i] = nx
        ch = True
    return o, ch
  if isinstance(o, dict):
    seen.add(id(o))
    ch = False
    for k, v in list(o.items()):
      nv, vc = _jax_k_rep_tree(v, pmap, drest, seen)
      if vc:
        drest.append((o, k, v))
        o[k] = nv
        ch = True
    return o, ch
  return o, False


def _jax_k_pipeline(
    a0: Any = None,
    a1: Any = None,
    a2: Any = None,
    a3: Any = None,
    buffering_level: int = 2,
    *,
    grid: Any = None,
    lower: Any = None,
    upper: Any = None,
    bounds: Any = None,
    init: Any = None,
    body: Any = None,
    disable_while_loop_copies: bool = True,
    has_dus: bool = False,
    is_nested: bool = False,
    tuple_1d: bool = False,
    scoped_vmem_limit_kib: Any = None,
    pipeline_loop_carried_dus: bool = False,
    immediate_revisiting: Any = None,
    async_pipelining_levels: Any = None,
    post_fusion_pipelined: bool = False,
    jump_fn: Any = None,
    pipeliner: Any = None,
    boundary_mode: Any = None,
    **kwargs: Any,
) -> Any:
  global _JAX_K_PAD_DISC
  del kwargs
  if pipeliner is not None:
    (
        pipeline_loop_carried_dus,
        immediate_revisiting,
        async_pipelining_levels,
        post_fusion_pipelined,
    ) = (
        pipeliner[k] if isinstance(pipeliner, dict) else getattr(pipeliner, k)
        for k in (
            "pipeline_loop_carried_dus",
            "immediate_revisiting",
            "async_pipelining_levels",
            "post_fusion_pipelined",
        )
    )
  b_m = dict(
      {
          "buffering_level": None,
          "xla_disable_while_loop_copies": None,
          "xla_while_loop_async_pipelining_levels": None,
          "xla_pipeline_loop_carried_dus": None,
          "xla_while_loop_async_immediate_revisiting": None,
          "xla_post_fusion_pipelined_loop": None,
      },
      **(
          {}
          if scoped_vmem_limit_kib is None
          else {"scoped_vmem_limit_kib": str(scoped_vmem_limit_kib)}
      ),
  )
  if a3 is not None:
    lower, upper, init, body = a0, a1, a2, a3
  elif a2 is not None:
    grid, init, body = a0, a1, a2
  elif a1 is not None and body is None and callable(a1):
    grid, body = a0, a1
  elif a1 is not None and init is not None:
    lower, upper = a0, a1
  elif a1 is not None:
    grid, init = a0, a1
  elif a0 is not None and grid is None and bounds is None and upper is None:
    grid = a0
  elif a0 is not None and lower is None:
    lower = a0
  b = grid if grid is not None else bounds
  if b is not None:
    if isinstance(b, (tuple, list)) and not b:
      return init
    is_seq = isinstance(b, (tuple, list))
    lo_t = (
        tuple(x[0] for x in b)
        if is_seq and isinstance(b[0], (tuple, list))
        else ((0,) * len(b) if is_seq else (0,))
    )
    hi_t = (
        tuple(x[1] for x in b)
        if is_seq and isinstance(b[0], (tuple, list))
        else (tuple(b) if is_seq else (b,))
    )
    is_t = is_seq and (
        len(b) > 1 or isinstance(b[0], (tuple, list)) or tuple_1d
    )
  else:
    is_seq = isinstance(lower, (tuple, list)) or isinstance(
        upper, (tuple, list)
    )
    lo_t = (
        tuple(lower)
        if isinstance(lower, (tuple, list))
        else (
            (0,) * len(upper)
            if isinstance(upper, (tuple, list))
            else (0 if lower is None else lower,)
        )
    )
    hi_t = tuple(upper) if isinstance(upper, (tuple, list)) else (upper,)
    is_t = len(hi_t) > 1 or tuple_1d
  st = (
      functools.reduce(
          int.__mul__, (max(0, int(h) - int(l)) for l, h in zip(lo_t, hi_t)), 1
      )
      if all(
          isinstance(x, int) and not isinstance(x, bool) for x in (*lo_t, *hi_t)
      )
      else None
  )
  if st == 0:
    return init
  lo_d = tuple(
      jnp.int32(l)
      if isinstance(l, int) and not isinstance(l, bool)
      else jnp.asarray(l, dtype=jnp.int32)
      for l in lo_t
  )
  hi_d = tuple(
      jnp.int32(h)
      if isinstance(h, int) and not isinstance(h, bool)
      else (
          h
          if isinstance(h, jax.Array) and h.dtype == jnp.int32
          else jnp.asarray(h, dtype=jnp.int32)
      )
      for h in hi_t
  )
  s_arg = (
      tuple(jax.ShapeDtypeStruct((), jnp.int32) for _ in lo_t)
      if is_t
      else jax.ShapeDtypeStruct((), jnp.int32)
  )
  if _JAX_K_PAD_DISC:
    if init is not None:

      def _idw(iv: Any, cv: Any) -> Any:
        s_lvs, (c_lvs, c_def) = jax.tree.leaves(init), jax.tree.flatten(cv)
        for s, cl in zip(s_lvs, c_lvs):
          ci = _jax_k_get_cidx(s)
          if ci is not None:
            _jax_k_set_cidx(cl, ci)
        return body(iv, jax.tree.unflatten(c_def, c_lvs))

      try:
        jax.eval_shape(_idw, s_arg, init)
      except Exception:
        pass
    return init
  crest, drest, orig_sh = [], [], []
  run_disc = (
      (boundary_mode in (None, "pad_hbm"))
      and _JAX_K_LOOP_DEPTH == 0
      and init is not None
      and jump_fn is None
  )
  glt, slt = globals().get("_jax_k_get_ltag", lambda _: None), globals().get(
      "_jax_k_set_ltag", lambda v, _: v
  )
  if run_disc:
    i_lvs, i_def = jax.tree.flatten(init)
    _JAX_K_CARRY_REQS.clear()
    _JAX_K_CLOS_REQS.clear()
    _JAX_K_CARRY_IDS.clear()
    _JAX_K_SRC_MAP.clear()
    for ci, lv in enumerate(i_lvs):
      _jax_k_set_cidx(lv, ci)
      if hasattr(lv, "shape"):
        _JAX_K_CARRY_REQS[ci] = (
            tuple(int(d) for d in lv.shape),
            [int(d) for d in lv.shape],
        )
    for cl in _jax_k_collect_cells(body):
      try:
        for lv in jax.tree.leaves(cl.cell_contents):
          if hasattr(lv, "shape"):
            _JAX_K_SRC_MAP[id(lv)] = (lv, lv)
      except ValueError:
        pass
    _JAX_K_PAD_DISC = True
    try:

      def _dw(iv: Any, cv: Any) -> Any:
        clvs, cdef = jax.tree.flatten(cv)
        for ci, cl in enumerate(clvs):
          _jax_k_set_cidx(cl, ci)
        return body(iv, jax.tree.unflatten(cdef, clvs))

      jax.eval_shape(_dw, s_arg, init)
    except Exception:
      pass
    finally:
      _JAX_K_PAD_DISC = False
    pmap, p_lvs = {}, []
    for ci, lv in enumerate(i_lvs):
      os_t = getattr(lv, "shape", ())
      en = _JAX_K_CARRY_REQS.get(ci)
      if en is not None and tuple(en[1]) != os_t:
        l_t = glt(lv)
        pl = jax.lax.pad(
            lv,
            jnp.zeros((), dtype=lv.dtype),
            tuple((0, max(0, r - d), 0) for d, r in zip(os_t, en[1])),
        )
        wl = globals().get("_jax_k_with_layout")
        if l_t is not None and wl is not None:
          pl = wl(pl, l_t)
        p_lvs.append(pl)
        orig_sh.append((True, os_t))
        pmap[id(lv)] = pl
      else:
        p_lvs.append(lv)
        orig_sh.append((False, os_t))
    for tid, (rt, os_t, rq) in _JAX_K_CLOS_REQS.items():
      if tuple(rq) != os_t and tid not in pmap:
        pmap[tid] = jax.lax.pad(
            rt,
            jnp.zeros((), dtype=rt.dtype),
            tuple((0, max(0, r - d), 0) for d, r in zip(os_t, rq)),
        )
    if pmap:
      for cl in _jax_k_collect_cells(body):
        try:
          cc = cl.cell_contents
        except ValueError:
          continue
        nc, ch = _jax_k_rep_tree(cc, pmap, drest, set())
        if ch:
          crest.append((cl, cc))
          cl.cell_contents = nc
      init = jax.tree.unflatten(i_def, p_lvs)
    _JAX_K_CARRY_REQS.clear()
    _JAX_K_CLOS_REQS.clear()
    _JAX_K_CARRY_IDS.clear()
    _JAX_K_SRC_MAP.clear()
  if is_seq and jump_fn is None:
    g_tup = tuple(jnp.maximum(jnp.int32(0), h - l) for l, h in zip(lo_d, hi_d))
    if st is not None:
      total = jnp.int32(st)
    else:
      total = jnp.int32(1)
      for ext in g_tup:
        total = total * ext
    se0 = jnp.maximum(jnp.int32(1), g_tup[0]) if not is_t else None
    s0: Any = jnp.int32(0)
  else:
    g_tup, se0, total = (), None, jnp.int32(0)
    s0 = lo_d if is_t else lo_d[0]
  c_lts = [glt(lv) for lv in jax.tree.leaves(init)] if init is not None else []

  def _cond_fn(state: Any) -> jax.Array:
    with xla_metadata.set_xla_metadata(**{
        "buffering_level": None,
        "xla_disable_while_loop_copies": None,
        "xla_while_loop_async_pipelining_levels": None,
        "xla_pipeline_loop_carried_dus": None,
        "xla_while_loop_async_immediate_revisiting": None,
        "xla_post_fusion_pipelined_loop": None,
    }):
      if is_seq and jump_fn is None:
        return state[0] < total
      return (
          jnp.all(jnp.stack([c < h for c, h in zip(state[0], hi_d)]))
          if is_t
          else state[0] < hi_d[0]
      )

  def _body_fn(state: Any) -> Any:
    global _JAX_K_LOOP_DEPTH
    with xla_metadata.set_xla_metadata(**b_m):
      i, carry, coords, rem = state[0], state[1], (), state[0]
      if jump_fn is not None:
        idx, nxt = i, jump_fn(i)
        nxt = (
            tuple(jnp.asarray(x, dtype=jnp.int32) for x in nxt)
            if is_t
            else jnp.asarray(nxt, dtype=jnp.int32)
        )
      elif is_seq and not is_t:
        idx, nxt = lo_d[0] + (i % se0), i + jnp.int32(1)
      elif is_t:
        for l_e, ext in reversed(list(zip(lo_d, g_tup))):
          se = jnp.maximum(jnp.int32(1), ext)
          coords, rem = (l_e + (rem % se), *coords), rem // se
        idx, nxt = coords, i + jnp.int32(1)
      else:
        idx, nxt = i, i + jnp.int32(1)
      _JAX_K_LOOP_DEPTH += 1
      try:
        nc = (body(idx) or jnp.int32(0)) if init is None else body(idx, carry)
      finally:
        _JAX_K_LOOP_DEPTH -= 1
      wl = globals().get("_jax_k_with_layout")
      if (
          init is not None
          and wl is not None
          and any(l is not None for l in c_lts)
      ):
        nlvs, ndef = jax.tree.flatten(nc)
        nc = jax.tree.unflatten(
            ndef,
            [
                wl(nlv, lt)
                if lt is not None and glt(nlv) is None and hasattr(nlv, "shape")
                else nlv
                for nlv, lt in zip(nlvs, c_lts)
            ],
        )
      return (nxt, nc)

  can_p = (st is None or st >= 4) and not is_nested and jump_fn is None
  mb = can_p and buffering_level >= 2
  sb = (
      can_p
      and buffering_level == 1
      and (
          not has_dus
          or pipeline_loop_carried_dus
          or async_pipelining_levels == 0
          or post_fusion_pipelined
      )
  )
  dc = disable_while_loop_copies and (mb or sb)
  w_m = {"xla_disable_while_loop_copies": "true"} if dc else {}
  if async_pipelining_levels is None and dc and (mb or not has_dus):
    async_pipelining_levels = buffering_level
  if pipeline_loop_carried_dus:
    w_m["xla_pipeline_loop_carried_dus"] = "true"
  if post_fusion_pipelined and dc:
    w_m["xla_post_fusion_pipelined_loop"] = "true"
  if immediate_revisiting is not None:
    w_m["xla_while_loop_async_immediate_revisiting"] = (
        "true" if immediate_revisiting else "false"
    )
  if async_pipelining_levels is not None:
    w_m["xla_while_loop_async_pipelining_levels"] = str(async_pipelining_levels)
  try:
    with xla_metadata.set_xla_metadata(**w_m):
      fc = (
          jax.lax.while_loop(_cond_fn, _body_fn, (s0, init))[1]
          if init is not None
          else (
              jax.lax.while_loop(_cond_fn, _body_fn, (s0, jnp.int32(0)))
              and None
          )
      )
  finally:
    for ct, k, ov in reversed(drest):
      ct[k] = ov
    for cl, ov in reversed(crest):
      cl.cell_contents = ov
  if fc is not None and any(l is not None for l in c_lts):
    for flv, lt in zip(jax.tree.leaves(fc), c_lts):
      if lt is not None:
        slt(flv, lt)
  if any(p for p, _ in orig_sh) and fc is not None:
    flvs, fdef = jax.tree.flatten(fc)
    fc = jax.tree.unflatten(
        fdef,
        [
            lv[tuple(slice(0, d) for d in os_t)] if p else lv
            for lv, (p, os_t) in zip(flvs, orig_sh)
        ],
    )
  return fc


# pylint: enable=invalid-name,missing-class-docstring,missing-function-docstring,broad-exception-caught,g-complex-comprehension


RCP_LN2 = 1.0 / math.log(2)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class KdaResiduals:
  """Prepared forward values retained when return_residuals is True."""

  q: jax.Array
  k: jax.Array
  v: jax.Array
  beta: jax.Array
  g_cumsum: jax.Array | None
  aqk: jax.Array
  akk: jax.Array
  initial_state: jax.Array | None
  g_org: jax.Array | None
  a_log: jax.Array | None
  delta_time_bias: jax.Array | None
  h: jax.Array | None
  g_dtype_marker: jax.Array
  q_rstd: jax.Array | None
  k_rstd: jax.Array | None
  cu_seqlens: jax.Array | None
  aligned_cu_seqlens: jax.Array | None
  chunk_indices: jax.Array | None
  aligned_segment_ids: jax.Array | None
  segment_ids: jax.Array | None
  cp_metadata: Any


def cdiv(x: Any, y: int) -> Any:
  return (x + y - 1) // y


def align_up(x: Any, align: int) -> Any:
  return (x + align - 1) // align * align


def _select(cond: Any, x: Any, y: Any) -> Any:
  """Selects between x and y using lax.select without jit(_where) MUST_FUSE."""
  dtype = jnp.result_type(x, y)
  x_arr = jnp.asarray(x, dtype=dtype)
  y_arr = jnp.asarray(y, dtype=dtype)
  shape = jnp.broadcast_shapes(cond.shape, x_arr.shape, y_arr.shape)
  return jax.lax.select(
      jnp.broadcast_to(cond, shape),
      jnp.broadcast_to(x_arr, shape),
      jnp.broadcast_to(y_arr, shape),
  )


def l2norm_fwd(x: Any, eps: Any = 1e-06) -> Any:
  """Normalizes x along the last dimension and returns (normalized, rstd)."""
  x_f = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + eps)
  return ((x_f * rstd[..., None]).astype(x.dtype), rstd.astype(jnp.float32))


def prepare_chunk_indices(
    cu_seqlens: Any, chunk_size: int, max_t: Any = None
) -> Any:
  """Computes per chunk (seq_id, block_id) mapping from cu_seqlens."""
  if cu_seqlens.ndim == 2:
    rows = [
        prepare_chunk_indices(cu_seqlens[b], chunk_size, max_t=max_t)
        for b in range(cu_seqlens.shape[0])
    ]
    return jnp.stack(rows, axis=0)
  lens = cu_seqlens[1:] - cu_seqlens[:-1]
  n_chunks = (lens + chunk_size - 1) // chunk_size
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


def segment_ids_to_seqlens(
    segment_ids: Any, max_segs: int, chunk_size: int = 1
) -> Any:
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
  out_idx = jnp.arange(max_segs + 1, dtype=jnp.int32)
  cu_seqlens = jnp.max(
      jnp.where(
          scatter_idx[None, :] == out_idx[:, None],
          scatter_val[None, :],
          jnp.int32(0),
      ),
      axis=1,
  )
  return jnp.where(out_idx > n_segs, n_real, cu_seqlens)


def segment_ids_to_cu_seqlens(
    segment_ids: Any, *, initial_state: Any, max_num_segments: Any
) -> Any:
  """Derives cu_seqlens and max_num_segments from segment_ids."""
  if segment_ids is None:
    return (None, max_num_segments)
  if max_num_segments is None:
    if initial_state is None:
      raise ValueError(
          "max_num_segments is required when segment_ids is provided without"
          " initial_state."
      )
    max_num_segments = initial_state.shape[1]
  return (
      segment_ids_to_seqlens(segment_ids, max_segs=max_num_segments),
      max_num_segments,
  )


def align_seqs(
    tensors_4d: Any,
    tensors_3d: Any,
    cu_seqlens: Any,
    align: int,
    aligned_cu_seqlens: Any = None,
    fill_values_4d: Any = None,
) -> Any:
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
          t4,
          t3,
          cu_seqlens[b],
          align,
          aligned_cu_seqlens=aligned_cu_b,
          fill_values_4d=fill_values_4d,
      )
      for idx, a in enumerate(aligned_4d):
        per_batch_4d[idx].append(a)
      for idx, a in enumerate(aligned_3d):
        per_batch_3d[idx].append(a)
      padded_cus.append(padded_cu_b)
      t_aligned_sizes.append(aligned_4d[0].shape[2])
    t_max = max(t_aligned_sizes)

    def _pad_and_cat_4d(tensors_per_batch: Any, fill_val: Any) -> Any:
      padded = []
      for t in tensors_per_batch:
        pad_len = t_max - t.shape[2]
        if pad_len > 0:
          t = jnp.pad(
              t,
              ((0, 0), (0, 0), (0, pad_len), (0, 0)),
              constant_values=jnp.asarray(fill_val, dtype=t.dtype),
          )
        padded.append(t)
      return jnp.concatenate(padded, axis=1)

    def _pad_and_cat_3d(tensors_per_batch: Any) -> Any:
      padded = []
      for t in tensors_per_batch:
        pad_len = t_max - t.shape[2]
        if pad_len > 0:
          t = jnp.pad(t, ((0, 0), (0, 0), (0, pad_len)))
        padded.append(t)
      return jnp.concatenate(padded, axis=1)

    fills = (
        fill_values_4d
        if fill_values_4d is not None
        else [0.0] * len(tensors_4d)
    )
    out_4d = [
        _pad_and_cat_4d(per_batch_4d[i], fills[i])
        for i in range(len(tensors_4d))
    ]
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
  block_align = 4 * align if t_old >= 8 * align else align
  t_new = (
      (t_old + n_segs * (align - 1) + block_align - 1)
      // block_align
      * block_align
  )
  j = jnp.arange(t_new, dtype=jnp.int32)[:, None]
  starts = padded_cu[:-1][None, :]
  lens = seg_lens[None, :]
  in_seg = (j >= starts) & (j < starts + lens)
  valid_mask = jnp.any(in_seg, axis=1)
  seg_id = jnp.argmax(in_seg, axis=1)
  gather_idx = jnp.where(
      valid_mask, cu_seqlens[seg_id] + (j[:, 0] - padded_cu[seg_id]), t_old
  )
  fills = (
      fill_values_4d if fill_values_4d is not None else [0.0] * len(tensors_4d)
  )

  def _gather_4d(t: Any, fill_val: Any) -> Any:
    fill_arr = jnp.asarray(fill_val, dtype=t.dtype)
    t_pad = jnp.pad(
        t, ((0, 0), (0, 0), (0, 1), (0, 0)), constant_values=fill_arr
    )
    return t_pad[:, :, gather_idx, :]

  def _gather_3d(t: Any) -> Any:
    t_pad = jnp.pad(t, ((0, 0), (0, 0), (0, 1)))
    return t_pad[:, :, gather_idx]

  with xla_metadata.set_xla_metadata(_xla_compute_type="dense"):
    out_4d = [_gather_4d(t, fills[idx]) for idx, t in enumerate(tensors_4d)]
    out_3d = [_gather_3d(t) for t in tensors_3d]
  return (out_4d, out_3d, padded_cu, cu_seqlens)


def unalign_output(
    o: Any, orig_cu_seqlens: Any, aligned_cu_seqlens: Any, t_out: int
) -> Any:
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
  t_aligned = o.shape[2]
  orig_seg_lens = orig_cu_seqlens[1:] - orig_cu_seqlens[:-1]
  j = jnp.arange(t_out, dtype=jnp.int32)[:, None]
  orig_starts = orig_cu_seqlens[:-1][None, :]
  lens = orig_seg_lens[None, :]
  in_seg = (j >= orig_starts) & (j < orig_starts + lens)
  valid_mask = jnp.any(in_seg, axis=1)
  seg_id = jnp.argmax(in_seg, axis=1)
  gather_idx = jnp.where(
      valid_mask,
      aligned_cu_seqlens[seg_id] + (j[:, 0] - orig_cu_seqlens[seg_id]),
      t_aligned,
  )
  with xla_metadata.set_xla_metadata(_xla_compute_type="dense"):
    o_pad = jnp.pad(o, ((0, 0), (0, 0), (0, 1), (0, 0)))
    return o_pad[:, :, gather_idx, :]


def _fused_dot(
    lhs: Any,
    rhs: Any,
    *,
    trans_a: Any = False,
    trans_b: Any = False,
    precision: Any = jax.lax.Precision.DEFAULT,
    rhs_scale: Any = None,
    scale: Any = None,
    add_to: Any = None,
    add_scale: Any = None,
    sub_from: Any = None,
    out_dtype: Any = jnp.float32,
) -> Any:
  """Executes a single MXU dot stage with optional fused epilogue scale, add, or sub."""
  lhs_v = lhs
  rhs_v = rhs
  if rhs_scale is not None:
    rhs_scale_v = rhs_scale

    def _dot_rhs_scale_fusion(a_in: Any, b_in: Any, b_scale_in: Any) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
          FUSE_LIMIT=None,
          scoped_vmem_limit_kib=None,
          buffering_level=None,
          output_window_bounds=None,
          kernel_window_bounds=None,
          input_window_bounds=None,
          integer=None,
          xla_disable_while_loop_copies=None,
          xla_while_loop_async_pipelining_levels=None,
          xla_pipeline_loop_carried_dus=None,
          xla_while_loop_async_immediate_revisiting=None,
          xla_post_fusion_pipelined_loop=None,
          xla_disjoint_read_write_regions=None,
          slice_sizes=None,
          ColorToVMEM=None,
          ColorInputToHBM=None,
          explicit_layout=None,
      ):
        b_eff = jnp.multiply(b_in, b_scale_in)
        prod = jnp.matmul(
            jnp.swapaxes(
                a_in.squeeze(0)
                if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_eff.ndim == 2)
                else a_in,
                -1,
                -2,
            )
            if trans_a
            else a_in.squeeze(0)
            if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_eff.ndim == 2)
            else a_in,
            jnp.swapaxes(
                b_eff.squeeze(0)
                if b_eff.ndim == 3 and b_eff.shape[0] == 1 and (a_in.ndim == 2)
                else b_eff,
                -1,
                -2,
            )
            if trans_b
            else b_eff.squeeze(0)
            if b_eff.ndim == 3 and b_eff.shape[0] == 1 and (a_in.ndim == 2)
            else b_eff,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return prod.astype(out_dtype)

    return _dot_rhs_scale_fusion(lhs_v, rhs_v, rhs_scale_v)
  if add_to is not None and add_scale is not None:
    add_v = add_to
    add_scale_v = add_scale

    def _dot_add_scaled_fusion(
        a_in: Any, b_in: Any, acc_in: Any, acc_scale_in: Any
    ) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
          FUSE_LIMIT=None,
          scoped_vmem_limit_kib=None,
          buffering_level=None,
          output_window_bounds=None,
          kernel_window_bounds=None,
          input_window_bounds=None,
          integer=None,
          xla_disable_while_loop_copies=None,
          xla_while_loop_async_pipelining_levels=None,
          xla_pipeline_loop_carried_dus=None,
          xla_while_loop_async_immediate_revisiting=None,
          xla_post_fusion_pipelined_loop=None,
          xla_disjoint_read_write_regions=None,
          slice_sizes=None,
          ColorToVMEM=None,
          ColorInputToHBM=None,
          explicit_layout=None,
      ):
        prod = jnp.matmul(
            jnp.swapaxes(
                a_in.squeeze(0)
                if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
                else a_in,
                -1,
                -2,
            )
            if trans_a
            else a_in.squeeze(0)
            if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
            else a_in,
            jnp.swapaxes(
                b_in.squeeze(0)
                if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
                else b_in,
                -1,
                -2,
            )
            if trans_b
            else b_in.squeeze(0)
            if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
            else b_in,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        acc_val = jnp.multiply(acc_in.astype(jnp.float32), acc_scale_in)
        return jnp.add(acc_val, prod).astype(out_dtype)

    return _dot_add_scaled_fusion(lhs_v, rhs_v, add_v, add_scale_v)
  if add_to is not None:
    add_v = add_to

    def _dot_add_fusion(a_in: Any, b_in: Any, acc_in: Any) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
          FUSE_LIMIT=None,
          scoped_vmem_limit_kib=None,
          buffering_level=None,
          output_window_bounds=None,
          kernel_window_bounds=None,
          input_window_bounds=None,
          integer=None,
          xla_disable_while_loop_copies=None,
          xla_while_loop_async_pipelining_levels=None,
          xla_pipeline_loop_carried_dus=None,
          xla_while_loop_async_immediate_revisiting=None,
          xla_post_fusion_pipelined_loop=None,
          xla_disjoint_read_write_regions=None,
          slice_sizes=None,
          ColorToVMEM=None,
          ColorInputToHBM=None,
          explicit_layout=None,
      ):
        prod = jnp.matmul(
            jnp.swapaxes(
                a_in.squeeze(0)
                if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
                else a_in,
                -1,
                -2,
            )
            if trans_a
            else a_in.squeeze(0)
            if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
            else a_in,
            jnp.swapaxes(
                b_in.squeeze(0)
                if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
                else b_in,
                -1,
                -2,
            )
            if trans_b
            else b_in.squeeze(0)
            if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
            else b_in,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return jnp.add(acc_in.astype(jnp.float32), prod).astype(out_dtype)

    return _dot_add_fusion(lhs_v, rhs_v, add_v)
  if sub_from is not None:
    sub_v = sub_from

    def _dot_sub_fusion(a_in: Any, b_in: Any, acc_in: Any) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
          FUSE_LIMIT=None,
          scoped_vmem_limit_kib=None,
          buffering_level=None,
          output_window_bounds=None,
          kernel_window_bounds=None,
          input_window_bounds=None,
          integer=None,
          xla_disable_while_loop_copies=None,
          xla_while_loop_async_pipelining_levels=None,
          xla_pipeline_loop_carried_dus=None,
          xla_while_loop_async_immediate_revisiting=None,
          xla_post_fusion_pipelined_loop=None,
          xla_disjoint_read_write_regions=None,
          slice_sizes=None,
          ColorToVMEM=None,
          ColorInputToHBM=None,
          explicit_layout=None,
      ):
        prod = jnp.matmul(
            jnp.swapaxes(
                a_in.squeeze(0)
                if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
                else a_in,
                -1,
                -2,
            )
            if trans_a
            else a_in.squeeze(0)
            if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
            else a_in,
            jnp.swapaxes(
                b_in.squeeze(0)
                if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
                else b_in,
                -1,
                -2,
            )
            if trans_b
            else b_in.squeeze(0)
            if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
            else b_in,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return jnp.subtract(acc_in.astype(jnp.float32), prod).astype(out_dtype)

    return _dot_sub_fusion(lhs_v, rhs_v, sub_v)
  if isinstance(scale, jax.Array):
    scale_v = scale

    def _dot_arr_scale_fusion(a_in: Any, b_in: Any, s_in: Any) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
          FUSE_LIMIT=None,
          scoped_vmem_limit_kib=None,
          buffering_level=None,
          output_window_bounds=None,
          kernel_window_bounds=None,
          input_window_bounds=None,
          integer=None,
          xla_disable_while_loop_copies=None,
          xla_while_loop_async_pipelining_levels=None,
          xla_pipeline_loop_carried_dus=None,
          xla_while_loop_async_immediate_revisiting=None,
          xla_post_fusion_pipelined_loop=None,
          xla_disjoint_read_write_regions=None,
          slice_sizes=None,
          ColorToVMEM=None,
          ColorInputToHBM=None,
          explicit_layout=None,
      ):
        prod = jnp.matmul(
            jnp.swapaxes(
                a_in.squeeze(0)
                if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
                else a_in,
                -1,
                -2,
            )
            if trans_a
            else a_in.squeeze(0)
            if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
            else a_in,
            jnp.swapaxes(
                b_in.squeeze(0)
                if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
                else b_in,
                -1,
                -2,
            )
            if trans_b
            else b_in.squeeze(0)
            if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
            else b_in,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return (prod * s_in.astype(jnp.float32)).astype(out_dtype)

    return _dot_arr_scale_fusion(lhs_v, rhs_v, scale_v)

  def _dot_stage_fusion(a_in: Any, b_in: Any) -> Any:
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=_jax_k_next_fuid(),
        FUSE_LIMIT=None,
        scoped_vmem_limit_kib=None,
        buffering_level=None,
        output_window_bounds=None,
        kernel_window_bounds=None,
        input_window_bounds=None,
        integer=None,
        xla_disable_while_loop_copies=None,
        xla_while_loop_async_pipelining_levels=None,
        xla_pipeline_loop_carried_dus=None,
        xla_while_loop_async_immediate_revisiting=None,
        xla_post_fusion_pipelined_loop=None,
        xla_disjoint_read_write_regions=None,
        slice_sizes=None,
        ColorToVMEM=None,
        ColorInputToHBM=None,
        explicit_layout=None,
    ):
      prod = jnp.matmul(
          jnp.swapaxes(
              a_in.squeeze(0)
              if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
              else a_in,
              -1,
              -2,
          )
          if trans_a
          else a_in.squeeze(0)
          if a_in.ndim == 3 and a_in.shape[0] == 1 and (b_in.ndim == 2)
          else a_in,
          jnp.swapaxes(
              b_in.squeeze(0)
              if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
              else b_in,
              -1,
              -2,
          )
          if trans_b
          else b_in.squeeze(0)
          if b_in.ndim == 3 and b_in.shape[0] == 1 and (a_in.ndim == 2)
          else b_in,
          precision=precision,
          preferred_element_type=jnp.float32,
      )
      if scale is not None:
        prod = prod * scale
      return prod.astype(out_dtype)

  return _dot_stage_fusion(lhs_v, rhs_v)


def _kda_intra_chunk_tile(
    q_tile: Any,
    k_tile: Any,
    v_tile: Any,
    g_tile: Any,
    beta_tile: Any,
    a_log_tile: Any,
    dt_bias_tile: Any,
    *,
    scale: Any,
    cumsum_scale: Any,
    safe_gate: Any,
    disable_recompute: Any,
    use_gate_in_kernel: Any,
    lower_bound: Any,
    compute_akk: Any = True,
    scale_qg: Any = False,
    high_prec: Any = False,
    compute_kg: Any = True,
    use_qk_l2norm: Any = False,
    return_packed_uw: Any = False,
) -> Any:
  """Computes Stage 1+2 over a 5D chunked tile (H, B, NC, BT, D) via staged MXU fusions."""
  orig_dtype = q_tile.dtype
  inter_dtype = jnp.float32 if high_prec else orig_dtype
  heads, batch, nc, bt, k_dim = q_tile.shape
  v_dim = v_tile.shape[-1]
  m_chunks = heads * batch * nc
  nc_sub = 4 if high_prec else 2
  bc = bt // nc_sub
  ref_idx = bc // 2 if safe_gate else 0
  solve_prec = (
      jax.lax.Precision.DEFAULT
      if orig_dtype == jnp.bfloat16
      else jax.lax.Precision.HIGHEST
  )
  cumsum_prec = solve_prec
  g_5d_v = g_tile
  a_log_v = (
      a_log_tile.astype(jnp.float32)
      if use_gate_in_kernel and a_log_tile is not None
      else None
  )
  dt_bias_v = (
      dt_bias_tile.astype(jnp.float32)
      if use_gate_in_kernel and dt_bias_tile is not None
      else None
  )

  def _cumsum_dot_fusion(g_in: Any, a_in: Any, dt_in: Any) -> Any:
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=_jax_k_next_fuid(),
        FUSE_LIMIT=None,
        scoped_vmem_limit_kib=None,
        buffering_level=None,
        output_window_bounds=None,
        kernel_window_bounds=None,
        input_window_bounds=None,
        integer=None,
        xla_disable_while_loop_copies=None,
        xla_while_loop_async_pipelining_levels=None,
        xla_pipeline_loop_carried_dus=None,
        xla_while_loop_async_immediate_revisiting=None,
        xla_post_fusion_pipelined_loop=None,
        xla_disjoint_read_write_regions=None,
        slice_sizes=None,
        ColorToVMEM=None,
        ColorInputToHBM=None,
        explicit_layout=None,
    ):
      g_f32 = g_in.astype(jnp.float32)
      if use_gate_in_kernel:
        if dt_in is not None:
          g_f32 = g_f32 + dt_in
        assert a_in is not None
        a_exp = jnp.exp(a_in)
        if lower_bound is None:
          g_f32 = -a_exp * jax.nn.softplus(g_f32)
        else:
          g_f32 = lower_bound * jax.nn.sigmoid(a_exp * g_f32)
      g_3d = g_f32.reshape(m_chunks, bt, k_dim)
      r_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 1)
      c_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 2)
      tril_3d = _select(r_3d >= c_3d, jnp.float32(1.0), jnp.float32(0.0))
      prod = jnp.matmul(
          tril_3d.squeeze(0)
          if tril_3d.ndim == 3 and tril_3d.shape[0] == 1 and (g_3d.ndim == 2)
          else tril_3d,
          g_3d.squeeze(0)
          if g_3d.ndim == 3 and g_3d.shape[0] == 1 and (tril_3d.ndim == 2)
          else g_3d,
          precision=cumsum_prec,
          preferred_element_type=jnp.float32,
      )
      return prod * cumsum_scale

  g_cumsum_3d = _cumsum_dot_fusion(g_5d_v, a_log_v, dt_bias_v)
  g_cumsum_5d = g_cumsum_3d.reshape(heads, batch, nc, bt, k_dim)
  g_cumsum_4d = g_cumsum_3d.reshape(m_chunks, nc_sub, bc, k_dim)
  q_3d_v = q_tile.reshape(m_chunks, bt, k_dim)
  k_3d_v = k_tile.reshape(m_chunks, bt, k_dim)
  if use_qk_l2norm:

    def _l2norm_dot_fusion(x_in: Any) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
          FUSE_LIMIT=None,
          scoped_vmem_limit_kib=None,
          buffering_level=None,
          output_window_bounds=None,
          kernel_window_bounds=None,
          input_window_bounds=None,
          integer=None,
          xla_disable_while_loop_copies=None,
          xla_while_loop_async_pipelining_levels=None,
          xla_pipeline_loop_carried_dus=None,
          xla_while_loop_async_immediate_revisiting=None,
          xla_post_fusion_pipelined_loop=None,
          xla_disjoint_read_write_regions=None,
          slice_sizes=None,
          ColorToVMEM=None,
          ColorInputToHBM=None,
          explicit_layout=None,
      ):
        x_2d = x_in.reshape(m_chunks * bt, k_dim)
        x_f = x_2d.astype(jnp.float32)
        x_sq = x_f * x_f
        ones_2d = jax.lax.broadcast_in_dim(jnp.float32(1.0), (k_dim, k_dim), ())
        sum_sq = jnp.matmul(
            x_sq.squeeze(0)
            if x_sq.ndim == 3 and x_sq.shape[0] == 1 and (ones_2d.ndim == 2)
            else x_sq,
            ones_2d.squeeze(0)
            if ones_2d.ndim == 3 and ones_2d.shape[0] == 1 and (x_sq.ndim == 2)
            else ones_2d,
            precision=solve_prec,
            preferred_element_type=jnp.float32,
        )
        rstd = jax.lax.rsqrt(sum_sq + 1e-06)
        return (x_f * rstd).astype(orig_dtype).reshape(m_chunks, bt, k_dim)

    q_norm_3d = _l2norm_dot_fusion(q_3d_v)
    k_norm_3d = _l2norm_dot_fusion(k_3d_v)
    q_3d_v = q_norm_3d
    k_3d_v = k_norm_3d
    q_4d_v = q_norm_3d.reshape(m_chunks, nc_sub, bc, k_dim)
    k_4d_v = k_norm_3d.reshape(m_chunks, nc_sub, bc, k_dim)
  else:
    q_4d_v = q_tile.reshape(m_chunks, nc_sub, bc, k_dim)
    k_4d_v = k_tile.reshape(m_chunks, nc_sub, bc, k_dim)
  v_3d_v = v_tile.reshape(m_chunks, bt, v_dim)
  beta_3d_v = beta_tile.reshape(m_chunks, bt, 1)
  beta_4d_v = beta_tile.reshape(m_chunks, nc_sub, bc, 1)
  g_cs_v = g_cumsum_3d
  g_cs_4d_v = g_cumsum_4d
  gn_ref_v = g_cumsum_4d[:, :, ref_idx : ref_idx + 1, :]
  g_last_v = g_cumsum_3d[:, bt - 1 : bt, :]

  def _k_eng_fusion(k_in: Any, g_in: Any, gn_ref_in: Any) -> Any:
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=_jax_k_next_fuid(),
        FUSE_LIMIT=None,
        scoped_vmem_limit_kib=None,
        buffering_level=None,
        output_window_bounds=None,
        kernel_window_bounds=None,
        input_window_bounds=None,
        integer=None,
        xla_disable_while_loop_copies=None,
        xla_while_loop_async_pipelining_levels=None,
        xla_pipeline_loop_carried_dus=None,
        xla_while_loop_async_immediate_revisiting=None,
        xla_post_fusion_pipelined_loop=None,
        xla_disjoint_read_write_regions=None,
        slice_sizes=None,
        ColorToVMEM=None,
        ColorInputToHBM=None,
        explicit_layout=None,
    ):
      k_full_4d = k_in.astype(jnp.float32)[:, None, :, :]
      g_full_4d = g_in[:, None, :, :]
      row_bt = jax.lax.broadcasted_iota(
          jnp.int32, (m_chunks, nc_sub, bt, k_dim), 2
      )
      sub_idx = jax.lax.broadcasted_iota(
          jnp.int32, (m_chunks, nc_sub, bt, k_dim), 1
      )
      valid_j = row_bt < (sub_idx + 1) * bc
      diff_j = _select(valid_j, gn_ref_in - g_full_4d, jnp.float32(0.0))
      k_eng = _select(valid_j, k_full_4d * jnp.exp2(diff_j), jnp.float32(0.0))
      return k_eng.astype(inter_dtype)

  k_eng_4d = _k_eng_fusion(k_3d_v, g_cs_v, gn_ref_v)
  k_eng_v = k_eng_4d

  def _aqk_dot_fusion(
      q_4d_in: Any, g_4d_in: Any, gn_ref_in: Any, k_eng_in: Any
  ) -> Any:
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=_jax_k_next_fuid(),
        FUSE_LIMIT=None,
        scoped_vmem_limit_kib=None,
        buffering_level=None,
        output_window_bounds=None,
        kernel_window_bounds=None,
        input_window_bounds=None,
        integer=None,
        xla_disable_while_loop_copies=None,
        xla_while_loop_async_pipelining_levels=None,
        xla_pipeline_loop_carried_dus=None,
        xla_while_loop_async_immediate_revisiting=None,
        xla_post_fusion_pipelined_loop=None,
        xla_disjoint_read_write_regions=None,
        slice_sizes=None,
        ColorToVMEM=None,
        ColorInputToHBM=None,
        explicit_layout=None,
    ):
      if high_prec:
        q_eg = (
            q_4d_in.astype(jnp.float32) * jnp.exp2(g_4d_in - gn_ref_in)
        ).astype(inter_dtype)
        prod = (
            jnp.matmul(
                q_eg.squeeze(0)
                if q_eg.ndim == 3
                and q_eg.shape[0] == 1
                and (k_eng_in.ndim == 2)
                else q_eg,
                jnp.swapaxes(
                    k_eng_in.squeeze(0)
                    if k_eng_in.ndim == 3
                    and k_eng_in.shape[0] == 1
                    and (q_eg.ndim == 2)
                    else k_eng_in,
                    -1,
                    -2,
                ),
                precision=solve_prec,
                preferred_element_type=jnp.float32,
            )
            * scale
        ).reshape(m_chunks, bt, bt)
      else:
        q_eg = (
            q_4d_in.astype(jnp.float32) * scale * jnp.exp2(g_4d_in - gn_ref_in)
        ).astype(inter_dtype)
        prod = jnp.matmul(
            q_eg.squeeze(0)
            if q_eg.ndim == 3 and q_eg.shape[0] == 1 and (k_eng_in.ndim == 2)
            else q_eg,
            jnp.swapaxes(
                k_eng_in.squeeze(0)
                if k_eng_in.ndim == 3
                and k_eng_in.shape[0] == 1
                and (q_eg.ndim == 2)
                else k_eng_in,
                -1,
                -2,
            ),
            precision=solve_prec,
            preferred_element_type=jnp.float32,
        ).reshape(m_chunks, bt, bt)
      r_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 1)
      c_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 2)
      return _select(r_3d >= c_3d, prod, jnp.float32(0.0)).astype(orig_dtype)

  def _l_dot_fusion(
      k_4d_in: Any, g_4d_in: Any, gn_ref_in: Any, beta_4d_in: Any, k_eng_in: Any
  ) -> Any:
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=_jax_k_next_fuid(),
        FUSE_LIMIT=None,
        scoped_vmem_limit_kib=None,
        buffering_level=None,
        output_window_bounds=None,
        kernel_window_bounds=None,
        input_window_bounds=None,
        integer=None,
        xla_disable_while_loop_copies=None,
        xla_while_loop_async_pipelining_levels=None,
        xla_pipeline_loop_carried_dus=None,
        xla_while_loop_async_immediate_revisiting=None,
        xla_post_fusion_pipelined_loop=None,
        xla_disjoint_read_write_regions=None,
        slice_sizes=None,
        ColorToVMEM=None,
        ColorInputToHBM=None,
        explicit_layout=None,
    ):
      if high_prec:
        k_eg = (
            k_4d_in.astype(jnp.float32) * jnp.exp2(g_4d_in - gn_ref_in)
        ).astype(inter_dtype)
        prod = (
            jnp.matmul(
                k_eg.squeeze(0)
                if k_eg.ndim == 3
                and k_eg.shape[0] == 1
                and (k_eng_in.ndim == 2)
                else k_eg,
                jnp.swapaxes(
                    k_eng_in.squeeze(0)
                    if k_eng_in.ndim == 3
                    and k_eng_in.shape[0] == 1
                    and (k_eg.ndim == 2)
                    else k_eng_in,
                    -1,
                    -2,
                ),
                precision=solve_prec,
                preferred_element_type=jnp.float32,
            )
            * beta_4d_in.astype(jnp.float32)
        ).reshape(m_chunks, bt, bt)
      else:
        k_eg = (
            k_4d_in.astype(jnp.float32)
            * beta_4d_in.astype(jnp.float32)
            * jnp.exp2(g_4d_in - gn_ref_in)
        ).astype(inter_dtype)
        prod = jnp.matmul(
            k_eg.squeeze(0)
            if k_eg.ndim == 3 and k_eg.shape[0] == 1 and (k_eng_in.ndim == 2)
            else k_eg,
            jnp.swapaxes(
                k_eng_in.squeeze(0)
                if k_eng_in.ndim == 3
                and k_eng_in.shape[0] == 1
                and (k_eg.ndim == 2)
                else k_eng_in,
                -1,
                -2,
            ),
            precision=solve_prec,
            preferred_element_type=jnp.float32,
        ).reshape(m_chunks, bt, bt)
      r_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 1)
      c_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 2)
      return _select(r_3d > c_3d, -prod, jnp.float32(0.0))

  if not high_prec and orig_dtype == jnp.float32:

    def _aqk_l_dot_fusion(
        q_4d_in: Any,
        k_4d_in: Any,
        g_4d_in: Any,
        gn_ref_in: Any,
        beta_4d_in: Any,
        k_eng_in: Any,
    ) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
          FUSE_LIMIT=None,
          scoped_vmem_limit_kib=None,
          buffering_level=None,
          output_window_bounds=None,
          kernel_window_bounds=None,
          input_window_bounds=None,
          integer=None,
          xla_disable_while_loop_copies=None,
          xla_while_loop_async_pipelining_levels=None,
          xla_pipeline_loop_carried_dus=None,
          xla_while_loop_async_immediate_revisiting=None,
          xla_post_fusion_pipelined_loop=None,
          xla_disjoint_read_write_regions=None,
          slice_sizes=None,
          ColorToVMEM=None,
          ColorInputToHBM=None,
          explicit_layout=None,
      ):
        exp_diff = jnp.exp2(g_4d_in - gn_ref_in)
        q_eg = q_4d_in * scale * exp_diff
        k_eg = k_4d_in * beta_4d_in * exp_diff
        qk_eg = jnp.concatenate([q_eg, k_eg], axis=2)
        prod = jnp.matmul(
            qk_eg.squeeze(0)
            if qk_eg.ndim == 3 and qk_eg.shape[0] == 1 and (k_eng_in.ndim == 2)
            else qk_eg,
            jnp.swapaxes(
                k_eng_in.squeeze(0)
                if k_eng_in.ndim == 3
                and k_eng_in.shape[0] == 1
                and (qk_eg.ndim == 2)
                else k_eng_in,
                -1,
                -2,
            ),
            precision=solve_prec,
            preferred_element_type=jnp.float32,
        )
        aqk_p = prod[:, :, :bc, :].reshape(m_chunks, bt, bt)
        l_p = prod[:, :, bc:, :].reshape(m_chunks, bt, bt)
        r_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 1)
        c_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 2)
        aqk_res = _select(r_3d >= c_3d, aqk_p, jnp.float32(0.0))
        neg_l_res = _select(r_3d > c_3d, -l_p, jnp.float32(0.0))
        return (aqk_res, neg_l_res)

    aqk_3d, neg_l = _aqk_l_dot_fusion(
        q_4d_v, k_4d_v, g_cs_4d_v, gn_ref_v, beta_4d_v, k_eng_v
    )
  else:
    aqk_3d = _aqk_dot_fusion(q_4d_v, g_cs_4d_v, gn_ref_v, k_eng_v)
    neg_l = _l_dot_fusion(k_4d_v, g_cs_4d_v, gn_ref_v, beta_4d_v, k_eng_v)
  packed_uw_5d: jax.Array | None = None
  if high_prec and orig_dtype == jnp.bfloat16:
    block_ids = jnp.arange(bt, dtype=jnp.int32) // 8
    same_block = (block_ids[:, None] == block_ids[None, :]).astype(jnp.float32)
    l_mat = -neg_l
    a_diag = l_mat * same_block[None]
    a_off_diag = l_mat - a_diag
    eye_bt = jnp.eye(bt, dtype=jnp.float32)[None]
    neg_a_diag = -a_diag
    diag_inv = eye_bt + neg_a_diag
    power = neg_a_diag
    for _ in range(2):
      power = _fused_dot(power, power, precision=solve_prec)
      diag_inv = _fused_dot(diag_inv, eye_bt + power, precision=solve_prec)
    v_beta = v_3d_v.astype(jnp.float32) * beta_3d_v.astype(jnp.float32)
    k_eg_beta = (
        k_3d_v.astype(jnp.float32)
        * jnp.exp2(g_cs_v)
        * beta_3d_v.astype(jnp.float32)
    )
    off_and_rhs = jnp.concatenate([a_off_diag, v_beta, k_eg_beta], axis=-1)
    merged = _fused_dot(diag_inv, off_and_rhs, precision=solve_prec)
    trans_off = merged[:, :, :bt]
    trans_rhs = merged[:, :, bt:]
    neg_trans_off = -trans_off
    off_inv = eye_bt + neg_trans_off
    power = neg_trans_off
    for _ in range(2):
      power = _fused_dot(power, power, precision=solve_prec)
      off_inv = _fused_dot(off_inv, power, precision=solve_prec, add_to=off_inv)
    res_uw = _fused_dot(off_inv, trans_rhs, precision=solve_prec)
    u_3d = res_uw[:, :, :v_dim].astype(orig_dtype)
    w_3d = res_uw[:, :, v_dim:].astype(orig_dtype)
    akk_3d = (
        _fused_dot(
            off_inv, diag_inv, precision=solve_prec, out_dtype=orig_dtype
        )
        if compute_akk
        else None
    )
  else:
    power = _fused_dot(neg_l, neg_l, precision=solve_prec)
    neg_l_v = neg_l
    power_v = power

    def _neumann_step0_fusion(nl_in: Any, p_in: Any) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
          FUSE_LIMIT=None,
          scoped_vmem_limit_kib=None,
          buffering_level=None,
          output_window_bounds=None,
          kernel_window_bounds=None,
          input_window_bounds=None,
          integer=None,
          xla_disable_while_loop_copies=None,
          xla_while_loop_async_pipelining_levels=None,
          xla_pipeline_loop_carried_dus=None,
          xla_while_loop_async_immediate_revisiting=None,
          xla_post_fusion_pipelined_loop=None,
          xla_disjoint_read_write_regions=None,
          slice_sizes=None,
          ColorToVMEM=None,
          ColorInputToHBM=None,
          explicit_layout=None,
      ):
        r_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 1)
        c_3d = jax.lax.broadcasted_iota(jnp.int32, (m_chunks, bt, bt), 2)
        akk_0 = _select(r_3d == c_3d, jnp.float32(1.0), nl_in)
        prod = jnp.matmul(
            akk_0.squeeze(0)
            if akk_0.ndim == 3 and akk_0.shape[0] == 1 and (p_in.ndim == 2)
            else akk_0,
            p_in.squeeze(0)
            if p_in.ndim == 3 and p_in.shape[0] == 1 and (akk_0.ndim == 2)
            else p_in,
            precision=solve_prec,
            preferred_element_type=jnp.float32,
        )
        return jnp.add(akk_0, prod)

    akk_3d_val = _neumann_step0_fusion(neg_l_v, power_v)
    num_rem_steps = 4 if high_prec else 2
    for step_idx in range(num_rem_steps):
      power = _fused_dot(power, power, precision=solve_prec)
      is_last = step_idx == num_rem_steps - 1
      akk_3d_val = _fused_dot(
          akk_3d_val,
          power,
          precision=solve_prec,
          add_to=akk_3d_val,
          out_dtype=inter_dtype if is_last else jnp.float32,
      )
    akk_v = akk_3d_val
    if v_dim == k_dim and (not high_prec):

      def _uw_dot_fusion(
          akk_in: Any, v_in: Any, k_in: Any, g_cs_in: Any, beta_in: Any
      ) -> Any:
        with xla_metadata.set_xla_metadata(
            MUST_FUSE=None,
            MAXIMAL_FUSE=_jax_k_next_fuid(),
            FUSE_LIMIT=None,
            scoped_vmem_limit_kib=None,
            buffering_level=None,
            output_window_bounds=None,
            kernel_window_bounds=None,
            input_window_bounds=None,
            integer=None,
            xla_disable_while_loop_copies=None,
            xla_while_loop_async_pipelining_levels=None,
            xla_pipeline_loop_carried_dus=None,
            xla_while_loop_async_immediate_revisiting=None,
            xla_post_fusion_pipelined_loop=None,
            xla_disjoint_read_write_regions=None,
            slice_sizes=None,
            ColorToVMEM=None,
            ColorInputToHBM=None,
            explicit_layout=None,
        ):
          beta_f = beta_in.astype(jnp.float32)
          v_beta = (v_in.astype(jnp.float32) * beta_f).astype(orig_dtype)
          k_eg_beta = (
              k_in.astype(jnp.float32) * jnp.exp2(g_cs_in) * beta_f
          ).astype(orig_dtype)
          is_w = (
              jax.lax.broadcasted_iota(jnp.int32, (m_chunks, 2, bt, k_dim), 1)
              == 1
          )
          vw_rhs = _select(is_w, k_eg_beta[:, None], v_beta[:, None])
          akk_4d = jnp.broadcast_to(akk_in[:, None], (m_chunks, 2, bt, bt))
          return jnp.matmul(
              akk_4d.squeeze(0)
              if akk_4d.ndim == 3
              and akk_4d.shape[0] == 1
              and (vw_rhs.ndim == 2)
              else akk_4d,
              vw_rhs.squeeze(0)
              if vw_rhs.ndim == 3
              and vw_rhs.shape[0] == 1
              and (akk_4d.ndim == 2)
              else vw_rhs,
              precision=solve_prec,
              preferred_element_type=jnp.float32,
          ).astype(orig_dtype)

      uw_4d = _uw_dot_fusion(akk_v, v_3d_v, k_3d_v, g_cs_v, beta_3d_v)
      if return_packed_uw:
        packed_uw_5d = uw_4d.reshape(heads, batch, nc * 2, bt, k_dim)
        u_3d = uw_4d
        w_3d = uw_4d
      else:
        u_3d = uw_4d[:, 0]
        w_3d = uw_4d[:, 1]
    else:

      def _u_dot_fusion(akk_in: Any, v_in: Any, beta_in: Any) -> Any:
        with xla_metadata.set_xla_metadata(
            MUST_FUSE=None,
            MAXIMAL_FUSE=_jax_k_next_fuid(),
            FUSE_LIMIT=None,
            scoped_vmem_limit_kib=None,
            buffering_level=None,
            output_window_bounds=None,
            kernel_window_bounds=None,
            input_window_bounds=None,
            integer=None,
            xla_disable_while_loop_copies=None,
            xla_while_loop_async_pipelining_levels=None,
            xla_pipeline_loop_carried_dus=None,
            xla_while_loop_async_immediate_revisiting=None,
            xla_post_fusion_pipelined_loop=None,
            xla_disjoint_read_write_regions=None,
            slice_sizes=None,
            ColorToVMEM=None,
            ColorInputToHBM=None,
            explicit_layout=None,
        ):
          v_beta = v_in.astype(jnp.float32) * beta_in.astype(jnp.float32)
          return jnp.matmul(
              akk_in.squeeze(0)
              if akk_in.ndim == 3
              and akk_in.shape[0] == 1
              and (v_beta.ndim == 2)
              else akk_in,
              v_beta.squeeze(0)
              if v_beta.ndim == 3
              and v_beta.shape[0] == 1
              and (akk_in.ndim == 2)
              else v_beta,
              precision=solve_prec,
              preferred_element_type=jnp.float32,
          ).astype(orig_dtype)

      def _w_dot_fusion(
          akk_in: Any, k_in: Any, g_cs_in: Any, beta_in: Any
      ) -> Any:
        with xla_metadata.set_xla_metadata(
            MUST_FUSE=None,
            MAXIMAL_FUSE=_jax_k_next_fuid(),
            FUSE_LIMIT=None,
            scoped_vmem_limit_kib=None,
            buffering_level=None,
            output_window_bounds=None,
            kernel_window_bounds=None,
            input_window_bounds=None,
            integer=None,
            xla_disable_while_loop_copies=None,
            xla_while_loop_async_pipelining_levels=None,
            xla_pipeline_loop_carried_dus=None,
            xla_while_loop_async_immediate_revisiting=None,
            xla_post_fusion_pipelined_loop=None,
            xla_disjoint_read_write_regions=None,
            slice_sizes=None,
            ColorToVMEM=None,
            ColorInputToHBM=None,
            explicit_layout=None,
        ):
          k_eg_beta = (
              k_in.astype(jnp.float32)
              * jnp.exp2(g_cs_in)
              * beta_in.astype(jnp.float32)
          )
          return jnp.matmul(
              akk_in.squeeze(0)
              if akk_in.ndim == 3
              and akk_in.shape[0] == 1
              and (k_eg_beta.ndim == 2)
              else akk_in,
              k_eg_beta.squeeze(0)
              if k_eg_beta.ndim == 3
              and k_eg_beta.shape[0] == 1
              and (akk_in.ndim == 2)
              else k_eg_beta,
              precision=solve_prec,
              preferred_element_type=jnp.float32,
          ).astype(orig_dtype)

      u_3d = _u_dot_fusion(akk_v, v_3d_v, beta_3d_v)
      w_3d = _w_dot_fusion(akk_v, k_3d_v, g_cs_v, beta_3d_v)
    akk_3d = akk_3d_val.astype(orig_dtype) if compute_akk else None

  def _kg_fusion(k_in: Any, g_cs_in: Any, g_last_in: Any) -> Any:
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=_jax_k_next_fuid(),
        FUSE_LIMIT=None,
        scoped_vmem_limit_kib=None,
        buffering_level=None,
        output_window_bounds=None,
        kernel_window_bounds=None,
        input_window_bounds=None,
        integer=None,
        xla_disable_while_loop_copies=None,
        xla_while_loop_async_pipelining_levels=None,
        xla_pipeline_loop_carried_dus=None,
        xla_while_loop_async_immediate_revisiting=None,
        xla_post_fusion_pipelined_loop=None,
        xla_disjoint_read_write_regions=None,
        slice_sizes=None,
        ColorToVMEM=None,
        ColorInputToHBM=None,
        explicit_layout=None,
    ):
      k_f = k_in.astype(jnp.float32)
      return (k_f * jnp.exp2(g_last_in - g_cs_in)).astype(orig_dtype)

  if disable_recompute:

    def _qg_fusion(q_in: Any, g_cs_in: Any) -> Any:
      with xla_metadata.set_xla_metadata(
          MUST_FUSE=None,
          MAXIMAL_FUSE=_jax_k_next_fuid(),
          FUSE_LIMIT=None,
          scoped_vmem_limit_kib=None,
          buffering_level=None,
          output_window_bounds=None,
          kernel_window_bounds=None,
          input_window_bounds=None,
          integer=None,
          xla_disable_while_loop_copies=None,
          xla_while_loop_async_pipelining_levels=None,
          xla_pipeline_loop_carried_dus=None,
          xla_while_loop_async_immediate_revisiting=None,
          xla_post_fusion_pipelined_loop=None,
          xla_disjoint_read_write_regions=None,
          slice_sizes=None,
          ColorToVMEM=None,
          ColorInputToHBM=None,
          explicit_layout=None,
      ):
        q_f = q_in.astype(jnp.float32)
        if scale_qg:
          q_f = q_f * scale
          g_eff = jnp.maximum(g_cs_in, -126.0)
        else:
          g_eff = g_cs_in
        return (q_f * jnp.exp2(g_eff)).astype(orig_dtype)

    kg_3d = _kg_fusion(k_3d_v, g_cs_v, g_last_v)
    qg_3d = _qg_fusion(q_3d_v, g_cs_v)
    kg_5d: jax.Array | None = kg_3d.reshape(heads, batch, nc, bt, k_dim)
    qg_5d: jax.Array | None = qg_3d.reshape(heads, batch, nc, bt, k_dim)
  elif compute_kg:
    kg_3d = _kg_fusion(k_3d_v, g_cs_v, g_last_v)
    kg_5d = kg_3d.reshape(heads, batch, nc, bt, k_dim)
    qg_5d = None
  else:
    kg_5d = None
    qg_5d = None
  if packed_uw_5d is not None:
    w_5d = packed_uw_5d
    u_5d = packed_uw_5d
  else:
    w_5d = w_3d.reshape(heads, batch, nc, bt, k_dim)
    u_5d = u_3d.reshape(heads, batch, nc, bt, v_dim)
  aqk_5d = aqk_3d.reshape(heads, batch, nc, bt, bt)
  akk_5d = (
      akk_3d.reshape(heads, batch, nc, bt, bt) if akk_3d is not None else None
  )
  return (w_5d, u_5d, qg_5d, kg_5d, aqk_5d, akk_5d, g_cumsum_5d)


def kda_fwd_intra_grid(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    a_log: Any = None,
    delta_time_bias: Any = None,
    *,
    scale: Any = 1.0,
    chunk_size: int = 64,
    safe_gate: Any = True,
    disable_recompute: Any = False,
    cumsum_scale: Any = RCP_LN2,
    use_gate_in_kernel: Any = False,
    lower_bound: Any = None,
    compute_akk: Any = True,
) -> Any:
  """jax_k Stage 1+2 kernel for fused gate cumsum and intra chunk solve."""
  heads, batch, seq_len, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = chunk_size
  nc = seq_len // bt
  q_5d = q.reshape(heads, batch, nc, bt, k_dim)
  k_5d = k.reshape(heads, batch, nc, bt, k_dim)
  v_5d = v.reshape(heads, batch, nc, bt, v_dim)
  g_5d = g.reshape(heads, batch, nc, bt, k_dim)
  beta_5d = beta.reshape(heads, batch, nc, bt, 1)
  a_log_5d = (
      a_log.astype(jnp.float32).reshape(heads, 1, 1, 1, 1)
      if use_gate_in_kernel and a_log is not None
      else None
  )
  dt_bias_5d = (
      delta_time_bias.astype(jnp.float32).reshape(heads, 1, 1, 1, k_dim)
      if use_gate_in_kernel and delta_time_bias is not None
      else None
  )
  w_5d, u_5d, qg_5d, kg_5d, aqk_5d, akk_5d, g_cumsum_5d = _kda_intra_chunk_tile(
      q_5d,
      k_5d,
      v_5d,
      g_5d,
      beta_5d,
      a_log_5d,
      dt_bias_5d,
      scale=scale,
      cumsum_scale=cumsum_scale,
      safe_gate=safe_gate,
      disable_recompute=disable_recompute,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
      compute_akk=compute_akk,
      high_prec=True,
  )
  assert kg_5d is not None
  w_out = w_5d.reshape(heads, batch, seq_len, k_dim)
  u_out = u_5d.reshape(heads, batch, seq_len, v_dim)
  qg_out = (
      qg_5d.reshape(heads, batch, seq_len, k_dim) if qg_5d is not None else None
  )
  kg_out = kg_5d.reshape(heads, batch, seq_len, k_dim)
  aqk_out = aqk_5d.reshape(heads, batch, seq_len, bt)
  akk_out = (
      akk_5d.reshape(heads, batch, seq_len, bt) if akk_5d is not None else None
  )
  g_cumsum_out = g_cumsum_5d.reshape(heads, batch, seq_len, k_dim)
  return (w_out, u_out, qg_out, kg_out, aqk_out, akk_out, g_cumsum_out)


@functools.partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "scale",
        "safe_gate",
        "disable_recompute",
        "cumsum_scale",
        "use_gate_in_kernel",
        "lower_bound",
        "mini_batch",
    ],
)
def jax_k_kda_fwd_intra_fused(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    scale: Any,
    chunk_size: int = 64,
    safe_gate: Any = True,
    disable_recompute: Any = False,
    cumsum_scale: Any = RCP_LN2,
    a_log: Any = None,
    delta_time_bias: Any = None,
    use_gate_in_kernel: Any = False,
    lower_bound: Any = None,
    mini_batch: Any = None,
) -> Any:
  """Fuses gate cumsum with the intra chunk solve in jax_k."""
  del mini_batch
  return kda_fwd_intra_grid(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      delta_time_bias,
      scale=scale,
      chunk_size=chunk_size,
      safe_gate=safe_gate,
      disable_recompute=disable_recompute,
      cumsum_scale=cumsum_scale,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
      compute_akk=True,
  )


def kda_fwd_intra_fused(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    scale: Any,
    cu_seqlens: Any = None,
    chunk_size: int = 64,
    chunk_indices: Any = None,
    safe_gate: Any = True,
    disable_recompute: Any = False,
    cumsum_scale: Any = RCP_LN2,
    a_log: Any = None,
    delta_time_bias: Any = None,
    use_gate_in_kernel: Any = False,
    lower_bound: Any = None,
) -> Any:
  """Fused gate cumsum and intra chunk solve entrypoint in jax_k."""
  del cu_seqlens, chunk_indices
  assert chunk_size == 64, f"Expected chunk_size=64, got {chunk_size}"
  return jax_k_kda_fwd_intra_fused(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      scale=scale,
      chunk_size=chunk_size,
      safe_gate=safe_gate,
      disable_recompute=disable_recompute,
      cumsum_scale=cumsum_scale,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
  )


def kda_fwd_pre_process_grid(
    k: Any,
    w: Any,
    u: Any,
    gk: Any,
    chunk_to_seq: Any,
    last_seg_idx: Any,
    *,
    chunk_size: int = 64,
) -> Any:
  """jax_k kernel for CP forward affine summary (S_ext, M) on B=1."""
  heads, _, t_local, k_dim = k.shape
  v_dim = u.shape[-1]
  bt = chunk_size
  nt = t_local // bt
  k_4d = k.reshape(heads, nt, bt, k_dim)
  w_4d = w.reshape(heads, nt, bt, k_dim)
  u_4d = u.reshape(heads, nt, bt, v_dim)
  gk_last_3d = gk.reshape(heads, nt, bt, k_dim)[:, :, bt - 1, :]
  eye_k = jnp.eye(k_dim, dtype=jnp.float32)
  eye_k_v = eye_k[None, None]

  def _per_chunk_factors_fusion(
      k_in: Any, w_in: Any, u_in: Any, gk_last_in: Any, eye_in: Any
  ) -> Any:
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=_jax_k_next_fuid(),
        FUSE_LIMIT=None,
        scoped_vmem_limit_kib=None,
        buffering_level=None,
        output_window_bounds=None,
        kernel_window_bounds=None,
        input_window_bounds=None,
        integer=None,
        xla_disable_while_loop_copies=None,
        xla_while_loop_async_pipelining_levels=None,
        xla_pipeline_loop_carried_dus=None,
        xla_while_loop_async_immediate_revisiting=None,
        xla_post_fusion_pipelined_loop=None,
        xla_disjoint_read_write_regions=None,
        slice_sizes=None,
        ColorToVMEM=None,
        ColorInputToHBM=None,
        explicit_layout=None,
    ):
      k_f32 = k_in.astype(jnp.float32)
      w_f32 = w_in.astype(jnp.float32)
      u_f32 = u_in.astype(jnp.float32)
      decay = jnp.exp2(jnp.maximum(gk_last_in.astype(jnp.float32), -126.0))
      diag = decay[..., None] * eye_in
      kw = jnp.matmul(
          jnp.swapaxes(
              k_f32.squeeze(0)
              if k_f32.ndim == 3 and k_f32.shape[0] == 1 and (w_f32.ndim == 2)
              else k_f32,
              -1,
              -2,
          ),
          w_f32.squeeze(0)
          if w_f32.ndim == 3 and w_f32.shape[0] == 1 and (k_f32.ndim == 2)
          else w_f32,
          precision=jax.lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )
      m_c = jnp.subtract(diag, kw)
      ds_c = jnp.matmul(
          jnp.swapaxes(
              k_f32.squeeze(0)
              if k_f32.ndim == 3 and k_f32.shape[0] == 1 and (u_f32.ndim == 2)
              else k_f32,
              -1,
              -2,
          ),
          u_f32.squeeze(0)
          if u_f32.ndim == 3 and u_f32.shape[0] == 1 and (k_f32.ndim == 2)
          else u_f32,
          precision=jax.lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )
      return (m_c, ds_c)

  m_all_chunks, ds_all_chunks = _per_chunk_factors_fusion(
      k_4d, w_4d, u_4d, gk_last_3d, eye_k_v
  )
  active_mask = chunk_to_seq == last_seg_idx[0]
  h_init = jnp.zeros((heads, k_dim, v_dim), dtype=jnp.float32)
  m_init = jnp.broadcast_to(eye_k, (heads, k_dim, k_dim)).copy()

  def _scan_body(c_idx: Any, carry: Any) -> Any:
    h_acc, m_acc = carry
    is_active = active_mask[c_idx]
    m_c = m_all_chunks[:, c_idx]
    ds_c = ds_all_chunks[:, c_idx]
    h_new = (
        jnp.matmul(
            m_c,
            h_acc,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        + ds_c
    )
    m_new = jnp.matmul(
        m_c,
        m_acc,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    h_next = jnp.where(is_active, h_new, h_acc)
    m_next = jnp.where(is_active, m_new, m_acc)
    return (h_next, m_next)

  h_final, m_final = jax.lax.fori_loop(0, nt, _scan_body, (h_init, m_init))
  s_ext_out = h_final[:, None, :, :]
  m_out = m_final[:, None, :, :]
  return (s_ext_out, m_out)


def chunk_gated_delta_rule_fwd_h_pre_process(
    k: Any,
    w: Any,
    u: Any,
    gk: Any,
    cu_seqlens: Any,
    chunk_indices: Any = None,
    chunk_size: int = 64,
    use_exp2: Any = True,
) -> Any:
  """Builds the CP affine summary (S_ext, M) for the last rank local segment."""
  _, batch, t_local, k_dim = k.shape
  bt = chunk_size
  assert use_exp2
  assert k_dim <= 256
  assert t_local % bt == 0
  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, bt, max_t=t_local)
  if batch > 1:
    s_list, m_list = ([], [])
    for b in range(batch):
      ci_b = chunk_indices[b] if chunk_indices.ndim == 3 else chunk_indices
      cu_b = cu_seqlens[b] if cu_seqlens.ndim == 2 else cu_seqlens
      s_b, m_b = chunk_gated_delta_rule_fwd_h_pre_process(
          k=k[:, b : b + 1],
          w=w[:, b : b + 1],
          u=u[:, b : b + 1],
          gk=gk[:, b : b + 1],
          cu_seqlens=cu_b,
          chunk_indices=ci_b,
          chunk_size=bt,
          use_exp2=use_exp2,
      )
      s_list.append(s_b)
      m_list.append(m_b)
    return (jnp.concatenate(s_list, axis=1), jnp.concatenate(m_list, axis=1))
  if cu_seqlens.ndim == 2:
    cu_seqlens = cu_seqlens[0]
  if chunk_indices.ndim == 3:
    chunk_indices = chunk_indices[0]
  chunk_to_seq = chunk_indices[:, 0].astype(jnp.int32)
  seg_lens = jnp.diff(cu_seqlens)
  seg_indices = jnp.arange(seg_lens.shape[0], dtype=jnp.int32)
  real_last_seg_idx = jnp.maximum(
      jnp.max(jnp.where(seg_lens > 0, seg_indices, jnp.int32(-1))), jnp.int32(0)
  ).astype(jnp.int32)[None]
  return kda_fwd_pre_process_grid(
      k, w, u, gk, chunk_to_seq, real_last_seg_idx, chunk_size=bt
  )


def _pack_o_chunks(o_list: Any) -> Any:
  """Packs a list of (H, B, BT, V) chunk outputs into (H, B, C*BT, V) in VMEM."""
  if len(o_list) == 1:
    return o_list[0]
  res = jnp.concatenate(o_list, axis=2)
  return res


def _stage34_precomputed_chunk_step(
    b_h: Any,
    b_w: Any,
    b_qg: Any,
    b_u: Any,
    b_kg: Any,
    b_decay: Any,
    b_a: Any,
    *,
    bt: int,
    output_prec: Any,
    out_dtype: Any,
    keep_state_mask: Any = None,
    h0_reset: Any = None,
    is_bos_mask: Any = None,
    mask_wq: Any = False,
) -> Any:
  """Executes one chunk step of Stage 3+4 from precomputed w, qg, and decay."""
  h_cur = b_h.astype(jnp.float32)
  if h0_reset is not None and is_bos_mask is not None:
    h_cur = _select(
        is_bos_mask[None, :, None, None], h0_reset.astype(jnp.float32), h_cur
    )
  elif keep_state_mask is not None and (not mask_wq):
    h_cur = _select(
        keep_state_mask[None, :, None, None] > 0.0, h_cur, jnp.float32(0.0)
    )
  w_v = b_w
  qg_v = b_qg
  u_v = b_u
  h_v = h_cur
  keep_v = keep_state_mask if keep_state_mask is not None and mask_wq else None

  def _wq_h_fusion(w_in: Any, qg_in: Any, h_in: Any, keep_in: Any) -> Any:
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=_jax_k_next_fuid(),
        FUSE_LIMIT=None,
        scoped_vmem_limit_kib=None,
        buffering_level=None,
        output_window_bounds=None,
        kernel_window_bounds=None,
        input_window_bounds=None,
        integer=None,
        xla_disable_while_loop_copies=None,
        xla_while_loop_async_pipelining_levels=None,
        xla_pipeline_loop_carried_dus=None,
        xla_while_loop_async_immediate_revisiting=None,
        xla_post_fusion_pipelined_loop=None,
        xla_disjoint_read_write_regions=None,
        slice_sizes=None,
        ColorToVMEM=None,
        ColorInputToHBM=None,
        explicit_layout=None,
    ):
      wq = jnp.concatenate([w_in, qg_in], axis=2)
      if keep_in is not None:
        wq = _select(
            keep_in[None, :, None, None] > 0.0,
            wq,
            jnp.zeros((), dtype=wq.dtype),
        )
      return jnp.matmul(
          wq.squeeze(0)
          if wq.ndim == 3 and wq.shape[0] == 1 and (h_in.ndim == 2)
          else wq,
          h_in.squeeze(0)
          if h_in.ndim == 3 and h_in.shape[0] == 1 and (wq.ndim == 2)
          else h_in,
          precision=jax.lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32,
      )

  wq_h = _wq_h_fusion(w_v, qg_v, h_v, keep_v)
  v_new = u_v.astype(jnp.float32) - wq_h[:, :, :bt, :]
  o_inter = wq_h[:, :, bt:, :]
  o_chunk = _fused_dot(
      b_a, v_new, precision=output_prec, add_to=o_inter, out_dtype=out_dtype
  )
  h_next = _fused_dot(
      b_kg,
      v_new,
      trans_a=True,
      precision=jax.lax.Precision.HIGHEST,
      add_to=h_cur,
      add_scale=b_decay,
  )
  return (o_chunk, h_next, h_cur, v_new.astype(out_dtype))


def _stage34_single_chunk_step(
    b_h: Any,
    b_w: Any,
    b_u: Any,
    b_kg: Any,
    b_g: Any,
    b_q: Any,
    b_a: Any,
    *,
    bt: int,
    scale: Any,
    output_prec: Any,
    out_dtype: Any,
    keep_state_mask: Any = None,
    h0_reset: Any = None,
    is_bos_mask: Any = None,
) -> Any:
  """Executes one chunk step of Stage 3+4 via staged MXU fusions."""
  h_cur = b_h.astype(jnp.float32)
  if h0_reset is not None and is_bos_mask is not None:
    h_cur = _select(
        is_bos_mask[None, :, None, None], h0_reset.astype(jnp.float32), h_cur
    )
  elif keep_state_mask is not None:
    h_cur = _select(
        keep_state_mask[None, :, None, None] > 0.0, h_cur, jnp.float32(0.0)
    )
  v_new = _fused_dot(
      b_w.astype(jnp.float32),
      h_cur,
      precision=jax.lax.Precision.HIGHEST,
      sub_from=b_u,
  )
  g_f32 = b_g.astype(jnp.float32)
  g_ref_row = g_f32[..., 0:1, :]
  qg_c = b_q.astype(jnp.float32) * jnp.exp2(
      jnp.maximum(g_f32 - g_ref_row, -126.0)
  )
  h_scaled = (
      h_cur * jnp.exp2(jnp.maximum(g_ref_row[..., 0, :], -126.0))[..., :, None]
  )
  o_inter = _fused_dot(qg_c, h_scaled, precision=output_prec, scale=scale)
  row_bt = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
  col_bt = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
  a_c = _select(
      (row_bt >= col_bt)[None, None], b_a.astype(jnp.float32), jnp.float32(0.0)
  )
  o_chunk = _fused_dot(
      a_c, v_new, precision=output_prec, add_to=o_inter, out_dtype=out_dtype
  )
  decay_c = jnp.exp2(g_f32[..., bt - 1, :])[..., :, None]
  h_next = _fused_dot(
      b_kg.astype(jnp.float32),
      v_new,
      trans_a=True,
      precision=jax.lax.Precision.HIGHEST,
      add_to=h_cur,
      add_scale=decay_c,
  )
  return (o_chunk, h_next, h_cur, v_new.astype(out_dtype))


def _stage34_block_steps(
    h_init: Any,
    w_blk: Any,
    qg_blk: Any,
    u_blk: Any,
    kg_blk: Any,
    decay_blk: Any,
    a_blk: Any,
    *,
    bt: int,
    output_prec: Any,
    out_dtype: Any,
) -> Any:
  """Steps through c_blk chunks in VMEM and returns (o_blk, h_final)."""
  heads, batch, t_blk, v_dim = u_blk.shape
  k_dim = kg_blk.shape[-1]
  c_blk = t_blk // bt
  if c_blk == 1:
    decay_c = decay_blk[:, :, 0, :, None]
    o_c, h_cur, _, _ = _stage34_precomputed_chunk_step(
        h_init,
        w_blk,
        qg_blk,
        u_blk,
        kg_blk,
        decay_c,
        a_blk,
        bt=bt,
        output_prec=output_prec,
        out_dtype=out_dtype,
    )
    return (o_c, h_cur)
  w_5d = w_blk.reshape(heads, batch, c_blk, bt, k_dim)
  qg_5d = qg_blk.reshape(heads, batch, c_blk, bt, k_dim)
  u_5d = u_blk.reshape(heads, batch, c_blk, bt, v_dim)
  kg_5d = kg_blk.reshape(heads, batch, c_blk, bt, k_dim)
  a_5d = a_blk[..., :bt].reshape(heads, batch, c_blk, bt, bt)
  h_cur = h_init
  o_list = []
  for idx in range(c_blk):
    decay_c = decay_blk[:, :, idx, :, None]
    o_c, h_cur, _, _ = _stage34_precomputed_chunk_step(
        h_cur,
        w_5d[:, :, idx],
        qg_5d[:, :, idx],
        u_5d[:, :, idx],
        kg_5d[:, :, idx],
        decay_c,
        a_5d[:, :, idx],
        bt=bt,
        output_prec=output_prec,
        out_dtype=out_dtype,
    )
    o_list.append(o_c)
  return (_pack_o_chunks(o_list), h_cur)


def _prepare_stage34_inputs(
    w: Any,
    q: Any,
    gk: Any,
    a: Any,
    keep_state: Any,
    *,
    bt: int,
    scale: Any,
    out_dtype: Any,
) -> Any:
  """Precomputes (w_eff, qg_eff, decay_4d, a_eff) outside the chunk loop."""
  heads, batch, seq_len, k_dim = q.shape
  nt = seq_len // bt
  q_v = q
  gk_v = gk

  def _qg_prep_fusion(q_in: Any, gk_in: Any) -> Any:
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=_jax_k_next_fuid(),
        FUSE_LIMIT=None,
        scoped_vmem_limit_kib=None,
        buffering_level=None,
        output_window_bounds=None,
        kernel_window_bounds=None,
        input_window_bounds=None,
        integer=None,
        xla_disable_while_loop_copies=None,
        xla_while_loop_async_pipelining_levels=None,
        xla_pipeline_loop_carried_dus=None,
        xla_while_loop_async_immediate_revisiting=None,
        xla_post_fusion_pipelined_loop=None,
        xla_disjoint_read_write_regions=None,
        slice_sizes=None,
        ColorToVMEM=None,
        ColorInputToHBM=None,
        explicit_layout=None,
    ):
      q_f = q_in.astype(jnp.float32) * scale
      gk_f = jnp.maximum(gk_in.astype(jnp.float32), -126.0)
      return (q_f * jnp.exp2(gk_f)).astype(out_dtype)

  qg_4d = _qg_prep_fusion(q_v, gk_v)
  gk_5d = gk.reshape(heads, batch, nt, bt, k_dim).astype(jnp.float32)
  decay_4d = jnp.exp2(gk_5d[:, :, :, bt - 1, :])
  if keep_state is not None:
    keep_5d = (keep_state > 0.0)[None, :, :, None, None]
    w_5d = jnp.where(
        keep_5d, w.reshape(heads, batch, nt, bt, k_dim).astype(out_dtype), 0
    )
    qg_5d = jnp.where(keep_5d, qg_4d.reshape(heads, batch, nt, bt, k_dim), 0)
    decay_4d = jnp.where(keep_5d[:, :, :, :, 0], decay_4d, 0.0)
    w_4d = w_5d.reshape(heads, batch, seq_len, k_dim)
    qg_4d = qg_5d.reshape(heads, batch, seq_len, k_dim)
  else:
    w_4d = w.astype(out_dtype)
  a_eff = a[..., :bt].astype(out_dtype)
  return (w_4d, qg_4d, decay_4d, a_eff)


def _stage34_wq_block_steps(
    h_init: Any,
    wq_blk: Any,
    u_blk: Any,
    kg_blk: Any,
    decay_blk: Any,
    a_blk: Any,
    *,
    bt: int,
    output_prec: Any,
    out_dtype: Any,
) -> Any:
  """Steps through c_blk chunks from interleaved wq_blk for c_block_size >= 4."""
  heads, batch, t_blk, v_dim = u_blk.shape
  k_dim = kg_blk.shape[-1]
  c_blk = t_blk // bt
  wq_5d = wq_blk.reshape(heads, batch, c_blk, 2 * bt, k_dim)
  u_5d = u_blk.reshape(heads, batch, c_blk, bt, v_dim)
  kg_5d = kg_blk.reshape(heads, batch, c_blk, bt, k_dim)
  a_5d = a_blk[..., :bt].reshape(heads, batch, c_blk, bt, bt)
  h_cur = h_init
  o_list = []
  for idx in range(c_blk):
    decay_c = decay_blk[:, :, idx, :, None]
    wq_h = _fused_dot(
        wq_5d[:, :, idx].astype(jnp.float32),
        h_cur,
        precision=jax.lax.Precision.HIGHEST,
    )
    wh = wq_h[..., :bt, :]
    o_inter = wq_h[..., bt:, :]
    v_new = u_5d[:, :, idx].astype(jnp.float32) - wh
    o_c = _fused_dot(
        a_5d[:, :, idx].astype(jnp.float32),
        v_new,
        precision=output_prec,
        add_to=o_inter,
        out_dtype=out_dtype,
    )
    h_cur = _fused_dot(
        kg_5d[:, :, idx].astype(jnp.float32),
        v_new,
        trans_a=True,
        precision=jax.lax.Precision.HIGHEST,
        add_to=h_cur,
        add_scale=decay_c,
    )
    o_list.append(o_c)
  return (_pack_o_chunks(o_list), h_cur)


def kda_fwd_h_o_grid(
    w: Any,
    u: Any,
    kg: Any,
    gk: Any,
    q: Any,
    a: Any,
    h_init_in: Any = None,
    keep_state: Any = None,
    *,
    scale: Any = 1.0,
    chunk_size: int = 64,
    c_block_size: int = 1,
) -> Any:
  """Pipelined jax_k Stage 3+4 kernel over blocks of chunks."""
  heads, batch, seq_len, k_dim = q.shape
  v_dim = u.shape[-1]
  bt = chunk_size
  nt = seq_len // bt
  out_dtype = q.dtype
  output_prec = (
      jax.lax.Precision.DEFAULT
      if q.dtype == jnp.bfloat16
      else jax.lax.Precision.HIGHEST
  )
  w_eff, qg_eff, decay_4d, a_eff = _prepare_stage34_inputs(
      w, q, gk, a, keep_state, bt=bt, scale=scale, out_dtype=out_dtype
  )
  if h_init_in is not None:
    h_init = h_init_in.astype(jnp.float32)
  else:
    h_init = jnp.zeros((heads, batch, k_dim, v_dim), dtype=jnp.float32)
  if nt < 4:
    o_all, _ = _stage34_block_steps(
        h_init,
        w_eff,
        qg_eff,
        u,
        kg,
        decay_4d,
        a_eff,
        bt=bt,
        output_prec=output_prec,
        out_dtype=out_dtype,
    )
    return o_all
  eff_c_blk = min(c_block_size, 4)
  while nt // eff_c_blk < 4 and eff_c_blk > 1:
    eff_c_blk //= 2
  num_blocks = nt // eff_c_blk
  t_blk = eff_c_blk * bt
  if c_block_size >= 4:
    w_5d = w_eff.reshape(heads, batch, nt, bt, k_dim)
    qg_5d = qg_eff.reshape(heads, batch, nt, bt, k_dim)
    is_qg = jax.lax.broadcasted_iota(jnp.int32, (2, bt, 1), 0) == 1
    wq_4d = jnp.where(
        is_qg[None, None, None], qg_5d[:, :, :, None], w_5d[:, :, :, None]
    ).reshape(heads, batch, nt * 2 * bt, k_dim)
    a_5d = a_eff.reshape(heads, batch, nt, bt, bt)
    row_bt = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
    col_bt = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
    a_masked = (
        jnp.where((row_bt >= col_bt)[None, None, None], a_5d, 0)
        .astype(out_dtype)
        .reshape(heads, batch, seq_len, bt)
    )
    wq_t_blk = eff_c_blk * 2 * bt
    wq_hbm = wq_4d
    u_hbm_b = u
    kg_hbm_b = kg
    decay_hbm_b = decay_4d
    a_hbm_b = a_masked
    out_buf_b = jnp.zeros(u.shape, dtype=out_dtype)

    def _process_wq_block(blk_idx: int, carry: Any) -> Any:
      curr_out, h_prev = carry
      t_start = blk_idx * t_blk
      wq_start = blk_idx * wq_t_blk
      c_start = blk_idx * eff_c_blk
      tile_wq = _jax_k_block_slice(
          wq_hbm,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(wq_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, wq_t_blk, k_dim),
      )
      tile_u = _jax_k_block_slice(
          u_hbm_b,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(t_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, t_blk, v_dim),
      )
      tile_kg = _jax_k_block_slice(
          kg_hbm_b,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(t_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, t_blk, k_dim),
      )
      tile_decay = _jax_k_block_slice(
          decay_hbm_b,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(c_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, eff_c_blk, k_dim),
      )
      tile_a = _jax_k_block_slice(
          a_hbm_b,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(t_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, t_blk, bt),
      )
      tile_o, h_next = _stage34_wq_block_steps(
          h_prev,
          tile_wq,
          tile_u,
          tile_kg,
          tile_decay,
          tile_a,
          bt=bt,
          output_prec=output_prec,
          out_dtype=out_dtype,
      )
      next_out = _jax_k_block_update(curr_out, tile_o, (0, 0, t_start, 0))
      return (next_out, h_next)

    curr_out_b, _ = _jax_k_pipeline(
        grid=(num_blocks,),
        init=(out_buf_b, h_init),
        body=_process_wq_block,
        buffering_level=2,
    )
    return curr_out_b
  w_hbm = w_eff
  qg_hbm = qg_eff
  u_hbm = u
  kg_hbm = kg
  decay_hbm = decay_4d
  a_hbm = a_eff
  out_buf = jnp.zeros(u.shape, dtype=out_dtype)

  def _process_block(blk_idx: int, carry: Any) -> Any:
    curr_out, h_prev = carry
    t_start = blk_idx * t_blk
    c_start = blk_idx * eff_c_blk
    tile_w = _jax_k_block_slice(
        w_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, k_dim),
    )
    tile_qg = _jax_k_block_slice(
        qg_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, k_dim),
    )
    tile_u = _jax_k_block_slice(
        u_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, v_dim),
    )
    tile_kg = _jax_k_block_slice(
        kg_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, k_dim),
    )
    tile_decay = _jax_k_block_slice(
        decay_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(c_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, eff_c_blk, k_dim),
    )
    tile_a = _jax_k_block_slice(
        a_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, bt),
    )
    tile_o, h_next = _stage34_block_steps(
        h_prev,
        tile_w,
        tile_qg,
        tile_u,
        tile_kg,
        tile_decay,
        tile_a,
        bt=bt,
        output_prec=output_prec,
        out_dtype=out_dtype,
    )
    next_out = _jax_k_block_update(
        curr_out, tile_o, (0, 0, t_start, 0), in_pipelined_loop=False
    )
    return (next_out, h_next)

  curr_out, _ = _jax_k_pipeline(
      grid=(num_blocks,),
      init=(out_buf, h_init),
      body=_process_block,
      buffering_level=1,
      has_dus=True,
  )
  return curr_out


def kda_fwd_h_o_with_state_grid(
    w: Any,
    u: Any,
    kg: Any,
    gk: Any,
    q: Any,
    a: Any,
    h_init_in: Any = None,
    *,
    scale: Any = 1.0,
    chunk_size: int = 64,
) -> Any:
  """Pipelined jax_k Stage 3+4 kernel returning both (o, h_final)."""
  heads, batch, seq_len, k_dim = q.shape
  v_dim = u.shape[-1]
  bt = chunk_size
  nt = seq_len // bt
  out_dtype = q.dtype
  output_prec = (
      jax.lax.Precision.DEFAULT
      if q.dtype == jnp.bfloat16
      else jax.lax.Precision.HIGHEST
  )
  w_eff, qg_eff, decay_4d, a_eff = _prepare_stage34_inputs(
      w, q, gk, a, None, bt=bt, scale=scale, out_dtype=out_dtype
  )
  if h_init_in is not None:
    h_init = h_init_in.astype(jnp.float32)
  else:
    h_init = jnp.zeros((heads, batch, k_dim, v_dim), dtype=jnp.float32)
  if nt < 4:
    o_all, h_final = _stage34_block_steps(
        h_init,
        w_eff,
        qg_eff,
        u,
        kg,
        decay_4d,
        a_eff,
        bt=bt,
        output_prec=output_prec,
        out_dtype=out_dtype,
    )
    return (o_all, h_final)
  w_hbm = w_eff
  qg_hbm = qg_eff
  u_hbm = u
  kg_hbm = kg
  decay_hbm = decay_4d
  a_hbm = a_eff
  out_buf = jnp.zeros(u.shape, dtype=out_dtype)

  def _process_block(blk_idx: int, carry: Any) -> Any:
    curr_out, h_prev = carry
    t_start = blk_idx * bt
    tile_w = _jax_k_block_slice(
        w_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, bt, k_dim),
    )
    tile_qg = _jax_k_block_slice(
        qg_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, bt, k_dim),
    )
    tile_u = _jax_k_block_slice(
        u_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, bt, v_dim),
    )
    tile_kg = _jax_k_block_slice(
        kg_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, bt, k_dim),
    )
    tile_decay = _jax_k_block_slice(
        decay_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(blk_idx, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, 1, k_dim),
    )
    tile_a = _jax_k_block_slice(
        a_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, bt, bt),
    )
    tile_o, h_next = _stage34_block_steps(
        h_prev,
        tile_w,
        tile_qg,
        tile_u,
        tile_kg,
        tile_decay,
        tile_a,
        bt=bt,
        output_prec=output_prec,
        out_dtype=out_dtype,
    )
    next_out = _jax_k_block_update(
        curr_out, tile_o, (0, 0, t_start, 0), in_pipelined_loop=False
    )
    return (next_out, h_next)

  curr_out, h_final = _jax_k_pipeline(
      grid=(nt,),
      init=(out_buf, h_init),
      body=_process_block,
      buffering_level=1,
      has_dus=True,
  )
  return (curr_out, h_final)


@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "scale",
        "chunk_size",
        "store_h",
        "store_v_new",
        "store_intermediates",
        "mini_batch",
    ],
)
def chunk_kda_fwd_h_o_varlen(
    w: Any,
    u: Any,
    kg: Any,
    gk: Any,
    q: Any,
    a: Any,
    cu_seqlens: Any,
    chunk_indices: Any = None,
    initial_state: Any = None,
    output_final_state: Any = False,
    scale: Any = 1.0,
    chunk_size: int = 64,
    store_h: Any = False,
    store_v_new: Any = False,
    store_intermediates: Any = None,
    mini_batch: Any = None,
    A: Any = None,  # pylint: disable=invalid-name
) -> Any:
  """Fuses varlen state propagation with output projection in jax_k."""
  del mini_batch
  if A is not None:
    a = A
  if store_intermediates is not None:
    store_h = store_h or store_intermediates
    store_v_new = store_v_new or store_intermediates
  heads, batch, seq_len, k_dim = q.shape
  v_dim = u.shape[-1]
  bt = chunk_size
  nt = seq_len // bt
  if cu_seqlens.ndim == 1:
    cu_seqlens = jnp.broadcast_to(
        cu_seqlens[None, :], (batch, cu_seqlens.shape[0])
    )
  n_segs = cu_seqlens.shape[-1] - 1
  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, bt, max_t=seq_len)
  chunk_to_seq = chunk_indices[..., 0].astype(jnp.int32)
  if chunk_to_seq.ndim == 1:
    chunk_to_seq = jnp.broadcast_to(chunk_to_seq[None, :], (batch, nt))
  bos_bc = jnp.take_along_axis(cu_seqlens, chunk_to_seq, axis=1)
  eos_bc = jnp.take_along_axis(cu_seqlens, chunk_to_seq + 1, axis=1)
  t0_c = jnp.arange(nt, dtype=jnp.int32)[None, :] * bt
  is_bos_bc = t0_c == bos_bc
  is_eos_bc = (t0_c < eos_bc) & (t0_c + bt >= eos_bc)
  can_use_fast_block_pipeline = (
      not store_h
      and (not store_v_new)
      and (not output_final_state or n_segs == 1)
      and (initial_state is None or n_segs == 1)
  )
  if can_use_fast_block_pipeline:
    h_init_arg = (
        jnp.transpose(initial_state[:, 0], (1, 0, 2, 3)).astype(jnp.float32)
        if initial_state is not None
        else None
    )
    if output_final_state:
      o_out, h_final = kda_fwd_h_o_with_state_grid(
          w, u, kg, gk, q, a, h_init_arg, scale=scale, chunk_size=bt
      )
      ht_out = jnp.transpose(h_final, (1, 0, 2, 3))[:, None, :, :, :]
      return (o_out, ht_out, None, None)
    keep_arg = (
        (~is_bos_bc).astype(jnp.float32)
        if initial_state is None and n_segs > 1
        else None
    )
    o_out = kda_fwd_h_o_grid(
        w,
        u,
        kg,
        gk,
        q,
        a,
        h_init_arg,
        keep_arg,
        scale=scale,
        chunk_size=bt,
        c_block_size=1,
    )
    return (o_out, None, None, None)
  out_dtype = q.dtype
  output_prec = (
      jax.lax.Precision.DEFAULT
      if q.dtype == jnp.bfloat16
      else jax.lax.Precision.HIGHEST
  )
  w_5d = w.reshape(heads, batch, nt, bt, k_dim)
  u_5d = u.reshape(heads, batch, nt, bt, v_dim)
  kg_5d = kg.reshape(heads, batch, nt, bt, k_dim)
  gk_5d = gk.reshape(heads, batch, nt, bt, k_dim)
  q_5d = q.reshape(heads, batch, nt, bt, k_dim)
  a_5d = a.reshape(heads, batch, nt, bt, bt)
  h_init_zero = jnp.zeros((heads, batch, k_dim, v_dim), dtype=jnp.float32)
  o_init = jnp.zeros((heads, batch, nt, bt, v_dim), dtype=out_dtype)
  ht_init = (
      jnp.zeros((batch, n_segs, heads, k_dim, v_dim), dtype=jnp.float32)
      if output_final_state
      else jnp.zeros((1,), dtype=jnp.float32)
  )
  h_hist_init = (
      jnp.zeros((heads, batch, nt, k_dim, v_dim), dtype=out_dtype)
      if store_h
      else jnp.zeros((1,), dtype=out_dtype)
  )
  v_new_init = (
      jnp.zeros((heads, batch, nt, bt, v_dim), dtype=out_dtype)
      if store_v_new
      else jnp.zeros((1,), dtype=out_dtype)
  )

  def _varlen_step(c_idx: Any, carry: Any) -> Any:
    h_cur, o_acc, ht_acc, h_hist_acc, v_new_acc = carry
    seq_c = chunk_to_seq[:, c_idx]
    bos_c = is_bos_bc[:, c_idx]
    eos_c = is_eos_bc[:, c_idx]
    if initial_state is not None:
      h0_c_b = jnp.take_along_axis(
          initial_state, seq_c[:, None, None, None, None], axis=1
      )[:, 0]
      h0_c = jnp.transpose(h0_c_b, (1, 0, 2, 3))
      keep_c = None
    else:
      h0_c = None
      keep_c = (~bos_c).astype(jnp.float32)
    o_c, h_next, h_pre, v_new_c = _stage34_single_chunk_step(
        h_cur,
        w_5d[:, :, c_idx],
        u_5d[:, :, c_idx],
        kg_5d[:, :, c_idx],
        gk_5d[:, :, c_idx],
        q_5d[:, :, c_idx],
        a_5d[:, :, c_idx],
        bt=bt,
        scale=scale,
        output_prec=output_prec,
        out_dtype=out_dtype,
        keep_state_mask=keep_c,
        h0_reset=h0_c,
        is_bos_mask=bos_c if initial_state is not None else None,
    )
    o_acc = o_acc.at[:, :, c_idx].set(o_c)
    if store_h:
      h_hist_acc = h_hist_acc.at[:, :, c_idx].set(h_pre.astype(out_dtype))
    if store_v_new:
      v_new_acc = v_new_acc.at[:, :, c_idx].set(v_new_c)
    if output_final_state:
      h_next_bh = jnp.transpose(h_next, (1, 0, 2, 3))
      for b_i in range(batch):
        ht_b = ht_acc[b_i, seq_c[b_i]]
        ht_upd = jnp.where(eos_c[b_i], h_next_bh[b_i], ht_b)
        ht_acc = ht_acc.at[b_i, seq_c[b_i]].set(ht_upd)
    return (h_next, o_acc, ht_acc, h_hist_acc, v_new_acc)

  _, o_5d, ht_res, h_hist_res, v_new_res = jax.lax.fori_loop(
      0,
      nt,
      _varlen_step,
      (h_init_zero, o_init, ht_init, h_hist_init, v_new_init),
  )
  o_out = o_5d.reshape(heads, batch, seq_len, v_dim)
  if output_final_state:
    seq_lens = jnp.diff(cu_seqlens, axis=-1)
    empty_mask = seq_lens == 0
    fill_value = (
        initial_state
        if initial_state is not None
        else jnp.zeros((batch, n_segs, heads, k_dim, v_dim), dtype=jnp.float32)
    )
    ht_out = jnp.where(empty_mask[:, :, None, None, None], fill_value, ht_res)
  else:
    ht_out = None
  h_out = h_hist_res if store_h else None
  v_new_out = (
      v_new_res.reshape(heads, batch, seq_len, v_dim) if store_v_new else None
  )
  return (o_out, ht_out, h_out, v_new_out)


def _fused_all_stages_block(
    h_init: Any,
    q_blk: Any,
    k_blk: Any,
    v_blk: Any,
    g_blk: Any,
    beta_blk: Any,
    a_log_5d: Any,
    dt_bias_5d: Any,
    keep_blk: Any,
    *,
    bt: int,
    scale: Any,
    cumsum_scale: Any,
    safe_gate: Any,
    use_gate_in_kernel: Any,
    lower_bound: Any,
    use_qk_l2norm: Any = False,
) -> Any:
  """Executes Stage 1+2 and Stage 3+4 over a block of chunks in VMEM."""
  heads, batch, t_blk, k_dim = q_blk.shape
  v_dim = v_blk.shape[-1]
  c_blk = t_blk // bt
  out_dtype = q_blk.dtype
  output_prec = (
      jax.lax.Precision.DEFAULT
      if out_dtype == jnp.bfloat16
      else jax.lax.Precision.HIGHEST
  )
  q_5d = q_blk.reshape(heads, batch, c_blk, bt, k_dim)
  k_5d = k_blk.reshape(heads, batch, c_blk, bt, k_dim)
  v_5d = v_blk.reshape(heads, batch, c_blk, bt, v_dim)
  g_5d = g_blk.reshape(heads, batch, c_blk, bt, k_dim)
  beta_5d = beta_blk.reshape(heads, batch, c_blk, bt, 1)
  use_packed_uw = v_dim == k_dim
  w_5d, u_5d, qg_5d, kg_5d, aqk_5d, _, g_cumsum_5d = _kda_intra_chunk_tile(
      q_5d,
      k_5d,
      v_5d,
      g_5d,
      beta_5d,
      a_log_5d,
      dt_bias_5d,
      scale=scale,
      cumsum_scale=cumsum_scale,
      safe_gate=safe_gate,
      disable_recompute=True,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
      compute_akk=False,
      scale_qg=True,
      use_qk_l2norm=use_qk_l2norm,
      return_packed_uw=use_packed_uw,
  )
  assert qg_5d is not None and kg_5d is not None
  decay_4d = jnp.exp2(g_cumsum_5d[:, :, :, bt - 1, :])
  if c_blk == 1:
    decay_c = decay_4d[:, :, 0, :, None]
    keep_c = keep_blk[:, 0] if keep_blk is not None else None
    if keep_c is not None:
      decay_c = _select(
          keep_c[None, :, None, None] > 0.0, decay_c, jnp.float32(0.0)
      )
    u_c = u_5d[:, :, 0]
    w_c = w_5d[:, :, 1] if use_packed_uw else w_5d[:, :, 0]
    o_c, h_cur, _, _ = _stage34_precomputed_chunk_step(
        h_init,
        w_c,
        qg_5d[:, :, 0],
        u_c,
        kg_5d[:, :, 0],
        decay_c,
        aqk_5d[:, :, 0],
        bt=bt,
        output_prec=output_prec,
        out_dtype=out_dtype,
        keep_state_mask=keep_c,
        mask_wq=True,
    )
    return (o_c, h_cur)
  h_cur = h_init
  o_list = []
  for idx in range(c_blk):
    decay_c = decay_4d[:, :, idx, :, None]
    keep_c = keep_blk[:, idx] if keep_blk is not None else None
    if keep_c is not None:
      decay_c = _select(
          keep_c[None, :, None, None] > 0.0, decay_c, jnp.float32(0.0)
      )
    u_c = u_5d[:, :, 2 * idx] if use_packed_uw else u_5d[:, :, idx]
    w_c = w_5d[:, :, 2 * idx + 1] if use_packed_uw else w_5d[:, :, idx]
    o_c, h_cur, _, _ = _stage34_precomputed_chunk_step(
        h_cur,
        w_c,
        qg_5d[:, :, idx],
        u_c,
        kg_5d[:, :, idx],
        decay_c,
        aqk_5d[:, :, idx],
        bt=bt,
        output_prec=output_prec,
        out_dtype=out_dtype,
        keep_state_mask=keep_c,
        mask_wq=True,
    )
    o_list.append(o_c)
  return (_pack_o_chunks(o_list), h_cur)


def kda_fwd_fused_grid(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    a_log: Any = None,
    delta_time_bias: Any = None,
    h_init_in: Any = None,
    keep_state: Any = None,
    *,
    scale: Any = 1.0,
    chunk_size: int = 64,
    safe_gate: Any = True,
    cumsum_scale: Any = RCP_LN2,
    use_gate_in_kernel: Any = False,
    lower_bound: Any = None,
    c_block_size: int = 4,
    use_qk_l2norm: Any = False,
) -> Any:
  """Single pass jax_k kernel fusing Stage 1+2 and Stage 3+4 in VMEM."""
  heads, batch, seq_len, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = chunk_size
  nt = seq_len // bt
  out_dtype = q.dtype
  a_log_5d = (
      a_log.astype(jnp.float32).reshape(heads, 1, 1, 1, 1)
      if use_gate_in_kernel and a_log is not None
      else None
  )
  dt_bias_5d = (
      delta_time_bias.astype(jnp.float32).reshape(heads, 1, 1, 1, k_dim)
      if use_gate_in_kernel and delta_time_bias is not None
      else None
  )
  if h_init_in is not None:
    h_init = h_init_in.astype(jnp.float32)
  else:
    h_init = jnp.zeros((heads, batch, k_dim, v_dim), dtype=jnp.float32)
  if nt < 4:
    o_all, _ = _fused_all_stages_block(
        h_init,
        q,
        k,
        v,
        g,
        beta,
        a_log_5d,
        dt_bias_5d,
        keep_state,
        bt=bt,
        scale=scale,
        cumsum_scale=cumsum_scale,
        safe_gate=safe_gate,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        use_qk_l2norm=use_qk_l2norm,
    )
    return o_all
  if nt % 4 == 0 and nt // 4 >= 4 and (nt // 4 % 2 == 0):
    eff_c_blk = min(c_block_size, 4)
  elif nt % 3 == 0 and nt // 3 >= 4 and (nt // 3 % 2 == 0):
    eff_c_blk = min(c_block_size, 3)
  else:
    eff_c_blk = min(c_block_size, 4)
    while (nt % eff_c_blk != 0 or nt // eff_c_blk < 4) and eff_c_blk > 1:
      eff_c_blk //= 2
  num_blocks = nt // eff_c_blk
  t_blk = eff_c_blk * bt
  q_hbm = q
  k_hbm = k
  v_hbm = v
  g_hbm = g
  beta_4d_hbm = beta.reshape(heads, batch, nt, bt)
  keep_hbm = keep_state if keep_state is not None else None
  out_buf = jnp.zeros(v.shape, dtype=out_dtype)

  def _process_block(blk_idx: int, carry: Any) -> Any:
    curr_out, h_prev = carry
    t_start = blk_idx * t_blk
    c_start = blk_idx * eff_c_blk
    tile_q = _jax_k_block_slice(
        q_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, k_dim),
    )
    tile_k = _jax_k_block_slice(
        k_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, k_dim),
    )
    tile_v = _jax_k_block_slice(
        v_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, v_dim),
    )
    tile_g = _jax_k_block_slice(
        g_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, k_dim),
    )
    tile_beta = _jax_k_block_slice(
        beta_4d_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(c_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, eff_c_blk, bt),
    )
    tile_keep = (
        _jax_k_block_slice(
            keep_hbm,
            (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(c_start, dtype=jnp.int32),
            ),
            (batch, eff_c_blk),
        )
        if keep_hbm is not None
        else None
    )
    tile_o, h_next = _fused_all_stages_block(
        h_prev,
        tile_q,
        tile_k,
        tile_v,
        tile_g,
        tile_beta,
        a_log_5d,
        dt_bias_5d,
        tile_keep,
        bt=bt,
        scale=scale,
        cumsum_scale=cumsum_scale,
        safe_gate=safe_gate,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        use_qk_l2norm=use_qk_l2norm,
    )
    next_out = _jax_k_block_update(
        curr_out, tile_o, (0, 0, t_start, 0), in_pipelined_loop=False
    )
    return (next_out, h_next)

  curr_out, _ = _jax_k_pipeline(
      grid=(num_blocks,),
      init=(out_buf, h_init),
      body=_process_block,
      buffering_level=1,
      has_dus=True,
  )
  return curr_out


def kda_fwd_fused_with_state_grid(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    a_log: Any = None,
    delta_time_bias: Any = None,
    h_init_in: Any = None,
    *,
    scale: Any = 1.0,
    chunk_size: int = 64,
    safe_gate: Any = True,
    cumsum_scale: Any = RCP_LN2,
    use_gate_in_kernel: Any = False,
    lower_bound: Any = None,
    c_block_size: int = 4,
    use_qk_l2norm: Any = False,
) -> Any:
  """Single pass jax_k kernel fusing Stage 1+2 and Stage 3+4 and returning state."""
  heads, batch, seq_len, k_dim = q.shape
  v_dim = v.shape[-1]
  bt = chunk_size
  nt = seq_len // bt
  out_dtype = q.dtype
  a_log_5d = (
      a_log.astype(jnp.float32).reshape(heads, 1, 1, 1, 1)
      if use_gate_in_kernel and a_log is not None
      else None
  )
  dt_bias_5d = (
      delta_time_bias.astype(jnp.float32).reshape(heads, 1, 1, 1, k_dim)
      if use_gate_in_kernel and delta_time_bias is not None
      else None
  )
  if h_init_in is not None:
    h_init = h_init_in.astype(jnp.float32)
  else:
    h_init = jnp.zeros((heads, batch, k_dim, v_dim), dtype=jnp.float32)
  if nt < 4:
    o_all, h_final = _fused_all_stages_block(
        h_init,
        q,
        k,
        v,
        g,
        beta,
        a_log_5d,
        dt_bias_5d,
        None,
        bt=bt,
        scale=scale,
        cumsum_scale=cumsum_scale,
        safe_gate=safe_gate,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        use_qk_l2norm=use_qk_l2norm,
    )
    return (o_all, h_final)
  if nt % 4 == 0 and nt // 4 >= 4 and (nt // 4 % 2 == 0):
    eff_c_blk = min(c_block_size, 4)
  elif nt % 3 == 0 and nt // 3 >= 4 and (nt // 3 % 2 == 0):
    eff_c_blk = min(c_block_size, 3)
  else:
    eff_c_blk = min(c_block_size, 4)
    while (nt % eff_c_blk != 0 or nt // eff_c_blk < 4) and eff_c_blk > 1:
      eff_c_blk //= 2
  num_blocks = nt // eff_c_blk
  t_blk = eff_c_blk * bt
  if heads * batch * nt <= 128:
    output_prec = (
        jax.lax.Precision.DEFAULT
        if out_dtype == jnp.bfloat16
        else jax.lax.Precision.HIGHEST
    )
    q_5d = q.reshape(heads, batch, nt, bt, k_dim)
    k_5d = k.reshape(heads, batch, nt, bt, k_dim)
    v_5d = v.reshape(heads, batch, nt, bt, v_dim)
    g_5d = g.reshape(heads, batch, nt, bt, k_dim)
    beta_5d = beta.reshape(heads, batch, nt, bt, 1)
    w_5d, u_5d, qg_5d, kg_5d, aqk_5d, _, g_cumsum_5d = _kda_intra_chunk_tile(
        q_5d,
        k_5d,
        v_5d,
        g_5d,
        beta_5d,
        a_log_5d,
        dt_bias_5d,
        scale=scale,
        cumsum_scale=cumsum_scale,
        safe_gate=safe_gate,
        disable_recompute=True,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        compute_akk=False,
        scale_qg=True,
        use_qk_l2norm=use_qk_l2norm,
    )
    assert qg_5d is not None and kg_5d is not None
    decay_4d = jnp.exp2(g_cumsum_5d[:, :, :, bt - 1, :])
    w_hbm = w_5d.reshape(heads, batch, seq_len, k_dim)
    qg_hbm = qg_5d.reshape(heads, batch, seq_len, k_dim)
    u_hbm = u_5d.reshape(heads, batch, seq_len, v_dim)
    kg_hbm = kg_5d.reshape(heads, batch, seq_len, k_dim)
    decay_hbm = decay_4d
    a_hbm = aqk_5d.reshape(heads, batch, seq_len, bt)
    out_buf_s = jnp.zeros(v.shape, dtype=out_dtype)

    def _process_stage34_block(blk_idx: int, carry: Any) -> Any:
      curr_out, h_prev = carry
      t_start = blk_idx * t_blk
      c_start = blk_idx * eff_c_blk
      tile_w = _jax_k_block_slice(
          w_hbm,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(t_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, t_blk, k_dim),
      )
      tile_qg = _jax_k_block_slice(
          qg_hbm,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(t_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, t_blk, k_dim),
      )
      tile_u = _jax_k_block_slice(
          u_hbm,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(t_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, t_blk, v_dim),
      )
      tile_kg = _jax_k_block_slice(
          kg_hbm,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(t_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, t_blk, k_dim),
      )
      tile_decay = _jax_k_block_slice(
          decay_hbm,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(c_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, eff_c_blk, k_dim),
      )
      tile_a = _jax_k_block_slice(
          a_hbm,
          (
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
              jnp.asarray(t_start, dtype=jnp.int32),
              jnp.asarray(0, dtype=jnp.int32),
          ),
          (heads, batch, t_blk, bt),
      )
      tile_o, h_next = _stage34_block_steps(
          h_prev,
          tile_w,
          tile_qg,
          tile_u,
          tile_kg,
          tile_decay,
          tile_a,
          bt=bt,
          output_prec=output_prec,
          out_dtype=out_dtype,
      )
      next_out = _jax_k_block_update(
          curr_out, tile_o, (0, 0, t_start, 0), in_pipelined_loop=False
      )
      return (next_out, h_next)

    curr_out_s, h_final_s = _jax_k_pipeline(
        grid=(num_blocks,),
        init=(out_buf_s, h_init),
        body=_process_stage34_block,
        buffering_level=1,
        has_dus=True,
    )
    return (curr_out_s, h_final_s)
  q_hbm = q
  k_hbm = k
  v_hbm = v
  g_hbm = g
  beta_4d_hbm = beta.reshape(heads, batch, nt, bt)
  out_buf = jnp.zeros(v.shape, dtype=out_dtype)

  def _process_block(blk_idx: int, carry: Any) -> Any:
    curr_out, h_prev = carry
    t_start = blk_idx * t_blk
    c_start = blk_idx * eff_c_blk
    tile_q = _jax_k_block_slice(
        q_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, k_dim),
    )
    tile_k = _jax_k_block_slice(
        k_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, k_dim),
    )
    tile_v = _jax_k_block_slice(
        v_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, v_dim),
    )
    tile_g = _jax_k_block_slice(
        g_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(t_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, t_blk, k_dim),
    )
    tile_beta = _jax_k_block_slice(
        beta_4d_hbm,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(c_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (heads, batch, eff_c_blk, bt),
    )
    tile_o, h_next = _fused_all_stages_block(
        h_prev,
        tile_q,
        tile_k,
        tile_v,
        tile_g,
        tile_beta,
        a_log_5d,
        dt_bias_5d,
        None,
        bt=bt,
        scale=scale,
        cumsum_scale=cumsum_scale,
        safe_gate=safe_gate,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        use_qk_l2norm=use_qk_l2norm,
    )
    next_out = _jax_k_block_update(
        curr_out, tile_o, (0, 0, t_start, 0), in_pipelined_loop=False
    )
    return (next_out, h_next)

  curr_out, h_final = _jax_k_pipeline(
      grid=(num_blocks,),
      init=(out_buf, h_init),
      body=_process_block,
      buffering_level=1,
      has_dus=True,
  )
  return (curr_out, h_final)


def _merge_initial_state(
    s_ext_all: Any,
    m_all: Any,
    rank: Any,
    pre_num_ranks: Any,
    is_first_rank: Any,
) -> Any:
  """Rebuilds the current rank initial state from all gathered (S_ext, M)."""
  cp_size, heads, batch, k_dim, v_dim = s_ext_all.shape
  s_in = jnp.zeros((heads, batch, k_dim, v_dim), dtype=jnp.float32)
  if cp_size == 1:
    return s_in
  pre_num_arr = jnp.asarray(pre_num_ranks, dtype=jnp.int32)
  is_first_arr = jnp.asarray(is_first_rank, dtype=jnp.bool_)
  rank_arr = jnp.asarray(rank, dtype=jnp.int32)

  def _body(i: Any, s_in_carry: Any) -> Any:
    offset = jnp.int32(cp_size - 1) - jnp.asarray(i, jnp.int32)
    j = rank_arr - offset
    active = (
        (offset <= pre_num_arr)
        & jnp.logical_not(is_first_arr)
        & (j >= 0)
        & (j < cp_size)
    )
    j_safe = jnp.clip(j, 0, cp_size - 1)
    m_j = jax.lax.dynamic_index_in_dim(m_all, j_safe, axis=0, keepdims=False)
    s_j = jax.lax.dynamic_index_in_dim(
        s_ext_all, j_safe, axis=0, keepdims=False
    )
    s_new = jnp.einsum("hbkj,hbjv->hbkv", m_j, s_in_carry) + s_j
    return jnp.where(active, s_new, s_in_carry)

  return jax.lax.fori_loop(0, cp_size - 1, _body, s_in)


def _prepare_cp_initial_state(
    *,
    kg: Any,
    w: Any,
    u: Any,
    gk: Any,
    cu_seqlens: Any,
    chunk_indices: Any,
    context_parallel_metadata: Any,
    chunk_size: int,
) -> Any:
  """Builds the rank local initial state for context parallel forward."""
  heads, batch, _, k_dim = kg.shape
  v_dim = u.shape[-1]
  n_segs = cu_seqlens.shape[-1] - 1
  s_ext_local, m_local = chunk_gated_delta_rule_fwd_h_pre_process(
      k=kg,
      w=w,
      u=u,
      gk=gk,
      cu_seqlens=cu_seqlens,
      chunk_indices=chunk_indices,
      chunk_size=chunk_size,
      use_exp2=True,
  )
  axis_name = context_parallel_metadata.axis_name
  s_ext_all = jax.lax.all_gather(
      s_ext_local, axis_name=axis_name, axis=0, tiled=False
  )
  m_all = jax.lax.all_gather(m_local, axis_name=axis_name, axis=0, tiled=False)
  rank = jax.lax.axis_index(axis_name)
  pre_num = context_parallel_metadata.pre_num_ranks
  is_first = context_parallel_metadata.is_first_rank
  if batch > 1 and hasattr(pre_num, "ndim") and (pre_num.ndim > 0):
    pre_num_arr = jnp.asarray(pre_num)
    is_first_arr = jnp.asarray(is_first)
    s_in_list = []
    for b_idx in range(batch):
      s_in_list.append(
          _merge_initial_state(
              s_ext_all[:, :, b_idx : b_idx + 1],
              m_all[:, :, b_idx : b_idx + 1],
              rank,
              pre_num_arr[b_idx],
              is_first_arr[b_idx],
          )
      )
    s_in_first = jnp.concatenate(s_in_list, axis=1)
  else:
    s_in_first = _merge_initial_state(s_ext_all, m_all, rank, pre_num, is_first)
  s_in_first = jnp.transpose(s_in_first, (1, 0, 2, 3))
  return (
      jnp.zeros((batch, n_segs, heads, k_dim, v_dim), dtype=jnp.float32)
      .at[:, 0]
      .set(s_in_first)
  )


def chunk_kda_fwd_custom(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    a_log: Any = None,
    delta_time_bias: Any = None,
    scale: Any = None,
    initial_state: Any = None,
    output_final_state: Any = False,
    use_gate_in_kernel: Any = False,
    segment_ids: Any = None,
    safe_gate: Any = True,
    lower_bound: Any = None,
    disable_recompute: Any = True,
    context_parallel_metadata: Any = None,
    chunk_size: int = 64,
    return_residuals: Any = False,
    cu_seqlens: Any = None,
    aligned_cu_seqlens: Any = None,
    chunk_indices: Any = None,
    aligned_segment_ids: Any = None,
    q_rstd: Any = None,
    k_rstd: Any = None,
    use_qk_l2norm: Any = False,
) -> Any:
  """Runs KDA forward in jax_k on canonicalized inputs."""
  original_cu_seqlens = cu_seqlens
  cu_seqlens = aligned_cu_seqlens
  save_for_backward = return_residuals and disable_recompute
  _, batch, seq_len, k_dim = q.shape
  bt = chunk_size
  scale_val = 1.0 / math.sqrt(k_dim) if scale is None else float(scale)
  cp_active = context_parallel_metadata is not None and getattr(
      context_parallel_metadata, "is_cp_enabled", False
  )
  cp_metadata = None
  if cp_active:
    cp_metadata = (
        context_parallel_metadata.is_first_rank,
        context_parallel_metadata.is_last_rank,
        context_parallel_metadata.pre_num_ranks,
        context_parallel_metadata.post_num_ranks,
    )
  is_varlen = cu_seqlens is not None
  nt = seq_len // bt
  n_segs = cu_seqlens.shape[-1] - 1 if cu_seqlens is not None else 1
  can_use_fused_grid = (
      not return_residuals
      and (not cp_active)
      and (not output_final_state or n_segs == 1)
      and (initial_state is None or n_segs == 1)
  )
  if can_use_fused_grid:
    if initial_state is not None:
      init_4d = (
          initial_state[:, 0] if initial_state.ndim == 5 else initial_state
      )
      h_init_arg = jnp.transpose(init_4d, (1, 0, 2, 3)).astype(jnp.float32)
    else:
      h_init_arg = None
    if output_final_state:
      o, h_final = kda_fwd_fused_with_state_grid(
          q,
          k,
          v,
          g,
          beta,
          a_log,
          delta_time_bias,
          h_init_arg,
          scale=scale_val,
          chunk_size=bt,
          safe_gate=safe_gate,
          cumsum_scale=RCP_LN2,
          use_gate_in_kernel=use_gate_in_kernel,
          lower_bound=lower_bound,
          use_qk_l2norm=use_qk_l2norm,
      )
      final_state_4d = jnp.transpose(h_final, (1, 0, 2, 3))
      final_state = (
          final_state_4d[:, None, :, :, :] if is_varlen else final_state_4d
      )
    else:
      if is_varlen and n_segs > 1:
        assert cu_seqlens is not None
        cu_2d = (
            jnp.broadcast_to(cu_seqlens[None, :], (batch, cu_seqlens.shape[0]))
            if cu_seqlens.ndim == 1
            else cu_seqlens
        )
        ci = (
            prepare_chunk_indices(cu_2d, bt, max_t=seq_len)
            if chunk_indices is None
            else chunk_indices
        )
        chunk_to_seq = ci[..., 0].astype(jnp.int32)
        if chunk_to_seq.ndim == 1:
          chunk_to_seq = jnp.broadcast_to(chunk_to_seq[None, :], (batch, nt))
        bos_bc = jnp.take_along_axis(cu_2d, chunk_to_seq, axis=1)
        t0_c = jnp.arange(nt, dtype=jnp.int32)[None, :] * bt
        keep_arg: jax.Array | None = (t0_c != bos_bc).astype(jnp.float32)
      else:
        keep_arg = None
      o = kda_fwd_fused_grid(
          q,
          k,
          v,
          g,
          beta,
          a_log,
          delta_time_bias,
          h_init_arg,
          keep_arg,
          scale=scale_val,
          chunk_size=bt,
          safe_gate=safe_gate,
          cumsum_scale=RCP_LN2,
          use_gate_in_kernel=use_gate_in_kernel,
          lower_bound=lower_bound,
          c_block_size=4,
          use_qk_l2norm=use_qk_l2norm,
      )
      final_state = None
    output = o
    cu_seqlens = original_cu_seqlens
    if aligned_cu_seqlens is not None:
      if segment_ids is None:
        raise ValueError("Aligned varlen metadata requires segment_ids.")
      assert cu_seqlens is not None
      output = unalign_output(
          output, cu_seqlens, aligned_cu_seqlens, segment_ids.shape[1]
      )
    return ((output.astype(q.dtype), final_state), None)
  w, u, _, kg, aqk, akk, g_cumsum = kda_fwd_intra_grid(
      q,
      k,
      v,
      g,
      beta,
      a_log,
      delta_time_bias,
      scale=scale_val,
      chunk_size=bt,
      safe_gate=safe_gate,
      disable_recompute=save_for_backward,
      cumsum_scale=RCP_LN2,
      use_gate_in_kernel=use_gate_in_kernel,
      lower_bound=lower_bound,
      compute_akk=return_residuals,
  )
  if cp_active:
    assert cu_seqlens is not None and chunk_indices is not None
    initial_state = _prepare_cp_initial_state(
        kg=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        context_parallel_metadata=context_parallel_metadata,
        chunk_size=bt,
    )
  if is_varlen:
    stage34_cu_seqlens = cu_seqlens
    stage34_chunk_indices = chunk_indices
    stage34_initial_state = initial_state
  else:
    stage34_cu_seqlens = jnp.broadcast_to(
        jnp.asarray([0, seq_len], dtype=jnp.int32), (batch, 2)
    )
    block_ids = jnp.arange(seq_len // bt, dtype=jnp.int32)
    fixed_chunk_indices = jnp.stack(
        [jnp.zeros_like(block_ids), block_ids], axis=-1
    )
    stage34_chunk_indices = jnp.broadcast_to(
        fixed_chunk_indices[None, :, :], (batch, seq_len // bt, 2)
    )
    stage34_initial_state = (
        initial_state[:, None, ...]
        if initial_state is not None and initial_state.ndim == 4
        else initial_state
    )
  assert stage34_cu_seqlens is not None
  o, final_state, h, _ = chunk_kda_fwd_h_o_varlen(
      w=w,
      u=u,
      kg=kg,
      gk=g_cumsum,
      q=q,
      a=aqk,
      cu_seqlens=stage34_cu_seqlens,
      chunk_indices=stage34_chunk_indices,
      initial_state=stage34_initial_state,
      output_final_state=output_final_state,
      scale=scale_val,
      chunk_size=bt,
      store_h=save_for_backward,
      store_v_new=False,
  )
  if not is_varlen and final_state is not None:
    final_state = final_state[:, 0]
  output = o
  cu_seqlens = original_cu_seqlens
  if aligned_cu_seqlens is not None:
    if segment_ids is None:
      raise ValueError("Aligned varlen metadata requires segment_ids.")
    assert cu_seqlens is not None
    output = unalign_output(
        output, cu_seqlens, aligned_cu_seqlens, segment_ids.shape[1]
    )
  result_pair = (output.astype(q.dtype), final_state)
  if not return_residuals:
    return (result_pair, None)
  if not save_for_backward:
    h = None
    if use_gate_in_kernel:
      g_cumsum = None
  g_org = g if use_gate_in_kernel else None
  assert akk is not None
  residuals = KdaResiduals(
      q=q,
      k=k,
      v=v,
      beta=beta,
      g_cumsum=g_cumsum,
      aqk=aqk,
      akk=akk,
      initial_state=initial_state,
      g_org=g_org,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      h=h,
      g_dtype_marker=jnp.zeros((), dtype=g.dtype),
      q_rstd=q_rstd,
      k_rstd=k_rstd,
      cu_seqlens=cu_seqlens,
      aligned_cu_seqlens=aligned_cu_seqlens,
      chunk_indices=chunk_indices,
      aligned_segment_ids=aligned_segment_ids,
      segment_ids=segment_ids,
      cp_metadata=cp_metadata,
  )
  return (result_pair, residuals)


def kda_fwd_jax_k(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    *,
    a_log: Any = None,
    delta_time_bias: Any = None,
    scale: Any = None,
    initial_state: Any = None,
    output_final_state: Any = False,
    use_qk_l2norm: Any = False,
    use_gate_in_kernel: Any = False,
    segment_ids: Any = None,
    safe_gate: Any = True,
    lower_bound: Any = None,
    disable_recompute: Any = True,
    chunk_size: int = 64,
    max_num_segments: Any = None,
) -> Any:
  """End to end jax_k forward execution for Kimi Delta Attention."""
  _, _, _, k_dim = q.shape
  bt = chunk_size
  scale_val = float(k_dim ** (-0.5)) if scale is None else float(scale)
  cu_seqlens, max_num_segments = segment_ids_to_cu_seqlens(
      segment_ids,
      initial_state=initial_state,
      max_num_segments=max_num_segments,
  )
  fuse_l2norm_in_kernel = (
      use_qk_l2norm and (not output_final_state) and (initial_state is None)
  )
  if use_qk_l2norm and (not fuse_l2norm_in_kernel):
    q_normed, q_rstd = l2norm_fwd(q)
    k_normed, k_rstd = l2norm_fwd(k)
  else:
    q_normed, k_normed = (q, k)
    q_rstd = k_rstd = None
  aligned_cu_seqlens = None
  chunk_indices = None
  if cu_seqlens is None:
    q_prepared, k_prepared, v_aligned = (q_normed, k_normed, v)
    g_aligned, beta_aligned = (g, beta)
  else:
    g_fill = -10000.0 if use_gate_in_kernel else 0.0
    (
        [q_prepared, k_prepared, v_aligned, g_aligned],
        [beta_aligned],
        aligned_cu_seqlens,
        _,
    ) = align_seqs(
        [q_normed, k_normed, v, g],
        [beta],
        cu_seqlens,
        align=bt,
        fill_values_4d=[0.0, 0.0, 0.0, g_fill],
    )
    chunk_indices = prepare_chunk_indices(
        aligned_cu_seqlens, bt, max_t=q_prepared.shape[2]
    )
  initial_state_prepared = initial_state
  if initial_state is not None:
    if aligned_cu_seqlens is None:
      if initial_state.ndim == 5:
        initial_state_prepared = initial_state[:, 0]
    else:
      state_count = aligned_cu_seqlens.shape[-1] - 1
      if initial_state.shape[1] < state_count:
        initial_state_prepared = jnp.pad(
            initial_state,
            (
                (0, 0),
                (0, state_count - initial_state.shape[1]),
                (0, 0),
                (0, 0),
                (0, 0),
            ),
        )
  output_pair, _ = chunk_kda_fwd_custom(
      q_prepared,
      k_prepared,
      v_aligned,
      g_aligned,
      beta_aligned,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      scale=scale_val,
      initial_state=initial_state_prepared,
      output_final_state=output_final_state,
      use_gate_in_kernel=use_gate_in_kernel,
      segment_ids=segment_ids,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      disable_recompute=disable_recompute,
      chunk_size=bt,
      return_residuals=False,
      cu_seqlens=cu_seqlens,
      aligned_cu_seqlens=aligned_cu_seqlens,
      chunk_indices=chunk_indices,
      q_rstd=q_rstd,
      k_rstd=k_rstd,
      use_qk_l2norm=fuse_l2norm_in_kernel,
  )
  out_val, final_state = output_pair
  if final_state is not None and final_state.ndim == 4:
    final_state = final_state[:, None]
  return (out_val, final_state)
