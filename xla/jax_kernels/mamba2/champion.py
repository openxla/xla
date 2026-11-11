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

"""Mamba2 Structured State Space Duality (SSD) kernels in jax_k."""

import functools
import inspect
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


def _fused_dot(
    lhs: Any,
    rhs: Any,
    *,
    trans_a: Any = False,
    trans_b: Any = False,
    precision: Any = jax.lax.Precision.HIGHEST,
    scale: Any = None,
    add_to: Any = None,
    add_scale: Any = None,
) -> Any:
  """Executes a single MXU dot stage with optional fused epilogue scale and add."""
  lhs_v = lhs
  rhs_v = rhs
  scale_v = scale if scale is not None else None
  add_to_v = add_to if add_to is not None else None
  add_scale_v = add_scale if add_scale is not None else None

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
      if scale_v is not None:
        prod = jnp.multiply(prod, scale_v)
      if add_to_v is not None:
        add_val = add_to_v.astype(jnp.float32)
        if add_scale_v is not None:
          add_val = jnp.multiply(add_val, add_scale_v)
        return jnp.add(add_val, prod)
      return prod

  return _dot_stage_fusion(lhs_v, rhs_v)


def _ssd_chunk_block_tile(
    x_dt_tile: Any,
    a_dt_tile: Any,
    b_tile: Any,
    c_tile: Any,
    h_init: Any = None,
    dt_eff_tile: Any = None,
    d_scale: Any = None,
) -> Any:
  """Computes Mamba2 SSD over a block of chunks via staged MXU fusions.

  Args:
    x_dt_tile: Chunked input of shape (B, H, K, Q, P) (pre scaled by dt if
      dt_eff_tile is None, or raw unscaled x if dt_eff_tile is provided).
    a_dt_tile: Chunked discretization decay of shape (B, H, K, Q).
    b_tile: Chunked B projection of shape (B, K, Q, S).
    c_tile: Chunked C projection of shape (B, K, Q, S).
    h_init: Optional incoming state of shape (B, H, S, P) entering chunk 0.
    dt_eff_tile: Optional softplus(dt + dt_bias) of shape (B, H, K, Q) for in
      VMEM input scaling.
    d_scale: Optional per head skip weight of shape (1, H, 1, 1, 1).

  Returns:
    Tuple (y_tile, h_final) where y_tile has shape (B, H, K, Q, P) and
    h_final has shape (B, H, S, P).
  """
  orig_dtype = x_dt_tile.dtype
  bsz, n_heads, n_chunks, q_len, d_head = x_dt_tile.shape
  d_state = b_tile.shape[-1]
  prec = jax.lax.Precision.HIGHEST
  swap_kh = q_len >= 128
  swap_ps = swap_kh and n_chunks >= 64
  x_dot = (jnp.swapaxes(x_dt_tile, 1, 2) if swap_kh else x_dt_tile).astype(
      jnp.float32
  )
  a_f32 = (jnp.swapaxes(a_dt_tile, 1, 2) if swap_kh else a_dt_tile).astype(
      jnp.float32
  )
  b_dot = b_tile.astype(jnp.float32)
  c_dot = c_tile.astype(jnp.float32)
  dt_f32 = (
      (jnp.swapaxes(dt_eff_tile, 1, 2) if swap_kh else dt_eff_tile).astype(
          jnp.float32
      )
      if dt_eff_tile is not None
      else None
  )
  d_scale_v = (
      (jnp.swapaxes(d_scale, 1, 2) if swap_kh else d_scale)
      if d_scale is not None
      else None
  )
  m_intra = _fused_dot(c_dot, b_dot, trans_b=True, precision=prec)
  row_q = jax.lax.broadcasted_iota(jnp.int32, (q_len, q_len), 0)
  col_q = jax.lax.broadcasted_iota(jnp.int32, (q_len, q_len), 1)
  causal_q = row_q >= col_q
  ones_tril_q = jnp.where(causal_q, 1.0, 0.0)
  a_cumsum = _fused_dot(a_f32, ones_tril_q, trans_b=True, precision=prec)

  def _intra_chunk_output_fusion(
      m_intra_v: Any, a_cumsum_v: Any, x_v: Any
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
      log_d_qq = a_cumsum_v[..., :, None] - a_cumsum_v[..., None, :]
      d_causal_qq = jnp.where(
          causal_q, jnp.exp(jnp.where(causal_q, log_d_qq, 0.0)), 0.0
      )
      m_bcast = (
          m_intra_v[:, :, None, :, :]
          if swap_kh
          else m_intra_v[:, None, :, :, :]
      )
      k_intra_v = jnp.multiply(m_bcast, d_causal_qq)
      if dt_f32 is not None:
        dt_v = dt_f32
        x_dt_v = jnp.multiply(x_v, dt_v[..., None])
      else:
        x_dt_v = x_v
      y_intra_val = jnp.matmul(
          k_intra_v.squeeze(0)
          if k_intra_v.ndim == 3
          and k_intra_v.shape[0] == 1
          and (x_dt_v.ndim == 2)
          else k_intra_v,
          x_dt_v.squeeze(0)
          if x_dt_v.ndim == 3 and x_dt_v.shape[0] == 1 and (k_intra_v.ndim == 2)
          else x_dt_v,
          precision=prec,
          preferred_element_type=jnp.float32,
      )
      if d_scale_v is not None:
        y_intra_val = jnp.add(jnp.multiply(x_v, d_scale_v), y_intra_val)
      return y_intra_val

  y_intra = _intra_chunk_output_fusion(m_intra, a_cumsum, x_dot)

  def _chunk_state_update_fusion(
      b_v_4d: Any, a_cumsum_v4: Any, x_v4: Any
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
      b_bcast = (
          b_v_4d[:, :, None, :, :] if swap_kh else b_v_4d[:, None, :, :, :]
      )
      b_v = b_bcast
      decay_to_end = jnp.exp(a_cumsum_v4[..., -1:] - a_cumsum_v4)
      if dt_f32 is not None:
        dt_v4 = dt_f32
        decay_to_end = jnp.multiply(decay_to_end, dt_v4)
      tilde_x_v = jnp.multiply(decay_to_end[..., None], x_v4)
      if swap_ps:
        return jnp.matmul(
            jnp.swapaxes(
                tilde_x_v.squeeze(0)
                if tilde_x_v.ndim == 3
                and tilde_x_v.shape[0] == 1
                and (b_v.ndim == 2)
                else tilde_x_v,
                -1,
                -2,
            ),
            b_v.squeeze(0)
            if b_v.ndim == 3 and b_v.shape[0] == 1 and (tilde_x_v.ndim == 2)
            else b_v,
            precision=prec,
            preferred_element_type=jnp.float32,
        )
      return jnp.matmul(
          jnp.swapaxes(
              b_v.squeeze(0)
              if b_v.ndim == 3 and b_v.shape[0] == 1 and (tilde_x_v.ndim == 2)
              else b_v,
              -1,
              -2,
          ),
          tilde_x_v.squeeze(0)
          if tilde_x_v.ndim == 3 and tilde_x_v.shape[0] == 1 and (b_v.ndim == 2)
          else tilde_x_v,
          precision=prec,
          preferred_element_type=jnp.float32,
      )

  h_new_raw = _chunk_state_update_fusion(b_dot, a_cumsum, x_dot)
  h_new = jnp.swapaxes(h_new_raw, 1, 2) if swap_kh else h_new_raw
  a_chunk_end = a_cumsum[..., -1]
  a_chunk_sum = jnp.swapaxes(a_chunk_end, 1, 2) if swap_kh else a_chunk_end
  row_k = jax.lax.broadcasted_iota(jnp.int32, (n_chunks, n_chunks), 0)
  col_k = jax.lax.broadcasted_iota(jnp.int32, (n_chunks, n_chunks), 1)
  strict_lower_k = row_k > col_k
  ones_tril_k = jnp.where(row_k >= col_k, 1.0, 0.0)
  cs = _fused_dot(a_chunk_sum, ones_tril_k, trans_b=True, precision=prec)
  cs_shifted = cs - a_chunk_sum
  h_new_flat = h_new.reshape(bsz, n_heads, n_chunks, d_state * d_head)

  def _inter_chunk_state_scan_fusion(
      cs_v: Any, cs_shift_v: Any, h_new_flat_v: Any
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
      log_inter_kk = cs_shift_v[..., :, None] - cs_v[..., None, :]
      d_inter_v = jnp.where(
          strict_lower_k,
          jnp.exp(jnp.where(strict_lower_k, log_inter_kk, 0.0)),
          0.0,
      )
      return jnp.matmul(
          d_inter_v.squeeze(0)
          if d_inter_v.ndim == 3
          and d_inter_v.shape[0] == 1
          and (h_new_flat_v.ndim == 2)
          else d_inter_v,
          h_new_flat_v.squeeze(0)
          if h_new_flat_v.ndim == 3
          and h_new_flat_v.shape[0] == 1
          and (d_inter_v.ndim == 2)
          else h_new_flat_v,
          precision=prec,
          preferred_element_type=jnp.float32,
      )

  h_prev_flat = _inter_chunk_state_scan_fusion(cs, cs_shifted, h_new_flat)
  h_prev = h_prev_flat.reshape(
      bsz,
      n_heads,
      n_chunks,
      d_head if swap_ps else d_state,
      d_state if swap_ps else d_head,
  )
  if h_init is not None:
    init_decay = jnp.exp(cs_shifted)[..., None, None]
    h_init_aligned = jnp.swapaxes(h_init, -1, -2) if swap_ps else h_init
    h_prev = h_prev + init_decay * h_init_aligned[:, :, None, :, :]

  def _off_diagonal_output_fusion(
      c_v_4d: Any, h_prev_v: Any, a_cumsum_v6: Any, y_intra_v: Any
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
      c_v = c_v_4d[:, None, :, :, :]
      y_off_raw = jnp.matmul(
          c_v.squeeze(0)
          if c_v.ndim == 3 and c_v.shape[0] == 1 and (h_prev_v.ndim == 2)
          else c_v,
          jnp.swapaxes(
              h_prev_v.squeeze(0)
              if h_prev_v.ndim == 3
              and h_prev_v.shape[0] == 1
              and (c_v.ndim == 2)
              else h_prev_v,
              -1,
              -2,
          )
          if swap_ps
          else h_prev_v.squeeze(0)
          if h_prev_v.ndim == 3 and h_prev_v.shape[0] == 1 and (c_v.ndim == 2)
          else h_prev_v,
          precision=prec,
          preferred_element_type=jnp.float32,
      )
      if swap_kh:
        y_off_raw = jnp.swapaxes(y_off_raw, 1, 2)
      decay_scale_v = jnp.exp(a_cumsum_v6)[..., None]
      return jnp.add(y_intra_v, jnp.multiply(y_off_raw, decay_scale_v)).astype(
          orig_dtype
      )

  h_prev_dot = h_prev
  y_total_raw = _off_diagonal_output_fusion(
      c_dot, h_prev_dot, a_cumsum, y_intra
  )
  y_total = jnp.swapaxes(y_total_raw, 1, 2) if swap_kh else y_total_raw
  block_end_decay = jnp.exp(cs[..., -1:] - cs)[..., None, None]
  h_final = jnp.sum(block_end_decay * h_new, axis=2)
  if swap_ps:
    h_final = jnp.swapaxes(h_final, -1, -2)
  if h_init is not None:
    total_block_decay = jnp.exp(cs[..., -1])[..., None, None]
    h_final = h_final + total_block_decay * h_init
  return (y_total, h_final)


def ssd_grid(
    x_dt: Any,
    a_dt: Any,
    b: Any,
    c: Any,
    out: Any = None,
    n_block_size: int = 64,
    dt_eff: Any = None,
    d: Any = None,
) -> Any:
  """Mamba2 SSD chunked kernel in jax_k.

  Executes loop free staged MXU fusions when n_block_size >= C, or a 1D
  software pipelined grid over chunk blocks when n_block_size < C.

  Args:
    x_dt: Chunked input in HBM of shape (B, H, C, Q, P) (pre scaled by dt if
      dt_eff is None, or raw unscaled x if dt_eff is provided).
    a_dt: Chunked log decay in HBM of shape (B, H, C, Q).
    b: Chunked B state projection in HBM of shape (B, C, Q, S).
    c: Chunked C state projection in HBM of shape (B, C, Q, S).
    out: Optional destination buffer in HBM matching x_dt.
    n_block_size: Number of chunks processed per pipeline step.
    dt_eff: Optional chunked softplus(dt + dt_bias) in HBM of shape (B, H, C,
      Q).
    d: Optional per head skip weight in HBM of shape (H,).

  Returns:
    Chunked SSD output tensor in HBM of shape (B, H, C, Q, P).
  """
  bsz, n_heads, num_chunks, q_len, d_head = x_dt.shape
  d_state = b.shape[-1]
  d_scale = (
      d.astype(jnp.float32)[None, :, None, None, None]
      if d is not None
      else None
  )
  if n_block_size >= num_chunks:
    y_out, _ = _ssd_chunk_block_tile(
        x_dt, a_dt, b, c, h_init=None, dt_eff_tile=dt_eff, d_scale=d_scale
    )
    return y_out
  if out is None:
    out_buf = jnp.zeros(x_dt.shape, dtype=x_dt.dtype)
  else:
    out_buf = out
  h_init_zero = jnp.zeros((bsz, n_heads, d_state, d_head), dtype=jnp.float32)
  num_blocks = (num_chunks + n_block_size - 1) // n_block_size

  def process_chunk_block(blk_idx: int, carry: Any) -> Any:
    curr_out, h_prev = carry
    c_start = blk_idx * n_block_size
    tile_x = _jax_k_block_slice(
        x_dt,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(c_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (bsz, n_heads, n_block_size, q_len, d_head),
    )
    tile_a = _jax_k_block_slice(
        a_dt,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(c_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (bsz, n_heads, n_block_size, q_len),
    )
    tile_b = _jax_k_block_slice(
        b,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(c_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (bsz, n_block_size, q_len, d_state),
    )
    tile_c = _jax_k_block_slice(
        c,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(c_start, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        (bsz, n_block_size, q_len, d_state),
    )
    tile_dt = (
        _jax_k_block_slice(
            dt_eff,
            (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(c_start, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            (bsz, n_heads, n_block_size, q_len),
        )
        if dt_eff is not None
        else None
    )
    tile_y, h_next = _ssd_chunk_block_tile(
        tile_x,
        tile_a,
        tile_b,
        tile_c,
        h_init=h_prev,
        dt_eff_tile=tile_dt,
        d_scale=d_scale,
    )
    next_out = _jax_k_block_update(curr_out, tile_y, (0, 0, c_start, 0, 0))
    return (next_out, h_next)

  res_out, _ = _jax_k_pipeline(
      grid=(num_blocks,),
      init=(out_buf, h_init_zero),
      body=process_chunk_block,
      buffering_level=2,
  )
  return res_out


def ssd_jax_k(
    *,
    dt: Any,
    x: Any,
    b: Any,
    c: Any,
    log_a: Any,
    d: Any,
    dt_bias: Any,
    block_len: int = 64,
    n_block_size: int = 64,
) -> Any:
  """Computes Mamba2 Structured State Space Duality (SSD) using jax_k."""
  orig_dtype = x.dtype
  batch_size, seq_len, d_inner = x.shape
  n_heads = dt.shape[-1]
  d_head = d_inner // n_heads
  d_state = b.shape[-1]
  num_chunks = seq_len // block_len
  if block_len >= 128:
    x_bcdq = jax.lax.optimization_barrier(
        jnp.swapaxes(
            x.reshape(batch_size, num_chunks, block_len, d_inner), 2, 3
        )
    )
    x_bhcqp = jnp.transpose(
        x_bcdq.reshape(batch_size, num_chunks, n_heads, d_head, block_len),
        (0, 2, 1, 4, 3),
    )
  else:
    x_bhcqp = jnp.transpose(
        x.reshape(batch_size, num_chunks, block_len, n_heads, d_head),
        (0, 3, 1, 2, 4),
    )
  dt_bhcq = jnp.transpose(
      dt.reshape(batch_size, num_chunks, block_len, n_heads), (0, 3, 1, 2)
  )
  dt_eff_f32 = jax.nn.softplus(
      dt_bhcq.astype(jnp.float32)
      + dt_bias.astype(jnp.float32)[None, :, None, None]
  )
  dt_eff_bhcq = dt_eff_f32.astype(orig_dtype)
  a_dt_bhcq = (
      -jnp.exp(log_a.astype(jnp.float32))[None, :, None, None] * dt_eff_f32
  ).astype(orig_dtype)
  b_bcqs = b.reshape(batch_size, num_chunks, block_len, d_state)
  c_bcqs = c.reshape(batch_size, num_chunks, block_len, d_state)
  try:
    y_bhcqp = ssd_grid(
        x_bhcqp,
        a_dt_bhcq,
        b_bcqs,
        c_bcqs,
        n_block_size=n_block_size,
        dt_eff=dt_eff_bhcq,
        d=d,
    )
  except TypeError as e:
    raise TypeError(
        f"ssd_jax_k(block_len={block_len}, n_block_size={n_block_size}): {e}"
    ) from e
  if block_len >= 128:
    y_bcdq = jax.lax.optimization_barrier(
        jnp.transpose(y_bhcqp, (0, 2, 1, 4, 3)).reshape(
            batch_size, num_chunks, d_inner, block_len
        )
    )
    return jnp.swapaxes(y_bcdq, 2, 3).reshape(batch_size, seq_len, d_inner)
  return jnp.transpose(y_bhcqp, (0, 2, 3, 1, 4)).reshape(
      batch_size, seq_len, d_inner
  )
