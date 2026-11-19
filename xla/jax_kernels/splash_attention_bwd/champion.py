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

"""Champion jax_k implementation of Splash Attention backward pass for TPU.

Evaluates a three output sibling dot mega fusion on TPU TensorCore:
  dV = P.T @ dO
  dP = dO @ V.T
  dS = P * (dP - D_i)
  dK = dS.T @ Q
  dQ = dS @ K
Keeping attention probabilities P and softmax gradients dS entirely in on chip
vector registers and vector memory (VMEM).
"""

import dataclasses
import functools
import inspect
from typing import Any

import jax
from jax import lax
from jax.experimental import scheduling_groups as jax_sched
from jax.experimental import xla_metadata
import jax.numpy as jnp

from xla.jax_kernels.splash_attention_bwd import configs

# pylint: disable=invalid-name,missing-class-docstring,missing-function-docstring,broad-exception-caught,g-complex-comprehension,g-long-ternary
_JAX_K_FUSION_UID = 0


def _jax_k_next_fuid(off: int = 0) -> str:
  global _JAX_K_FUSION_UID
  _JAX_K_FUSION_UID += 1
  return str(off + _JAX_K_FUSION_UID)


def _jax_k_fmt_meta(v: Any) -> str:
  return (
      str(v).lower()
      if isinstance(v, bool)
      else (
          ','.join(str(x) for x in v)
          if isinstance(v, (tuple, list))
          else str(v)
      )
  )


def _jax_k_fusion_meta(
    *, must_fuse: bool = False, fusion_group: Any = None, **kw: Any
) -> dict[str, str]:
  out: dict[str, str] = {}
  if must_fuse:
    out['MUST_FUSE'] = (
        _jax_k_fmt_meta(fusion_group)
        if fusion_group is not None
        else _jax_k_next_fuid(1000)
    )
  for k, m in (
      ('scoped_vmem_limit_kib', 'scoped_vmem_limit_kib'),
      ('buffering_level', 'buffering_level'),
      ('output_window_bounds', 'output_window_bounds'),
      ('kernel_window_bounds', 'kernel_window_bounds'),
      ('input_window_bounds', 'input_window_bounds'),
      ('integer', 'integer'),
      ('fuse_limit', 'FUSE_LIMIT'),
  ):
    v = kw.get(k)
    if v is not None:
      out[m] = _jax_k_fmt_meta(v)
  return out


def _jax_k_fusion(
    fn: Any = None,
    vmem_kib: Any = 64,
    *,
    scoped_vmem_limit_kib: Any = None,
    buffering_level: Any = None,
    output_window_bounds: Any = None,
    window_bounds: Any = None,
    kernel_window_bounds: Any = None,
    input_window_bounds: Any = None,
    window_config: Any = None,
    integer: Any = None,
    must_fuse: bool = False,
    fusion_group: Any = None,
    unsafe_metadata: Any = None,
    _clear_extra: tuple[str, ...] = (),
    _flag_overrides: Any = None,
    _non_vmem_idxs: tuple[int, ...] = (),
    _non_vmem_names: tuple[str, ...] = (),
    _has_ret_layout: bool = False,
    **extra: Any,
) -> Any:
  owb = (
      output_window_bounds
      if output_window_bounds is not None
      else window_bounds
  )
  sv_exp = scoped_vmem_limit_kib is not None or (
      'scoped_vmem_limit_kib' in extra
      and extra['scoped_vmem_limit_kib'] is not None
  )
  sv_imp = (
      None
      if sv_exp
      else (vmem_kib if isinstance(vmem_kib, int) and vmem_kib >= 512 else None)
  )
  base_cm: dict[str, str] = {}
  for k, v in (
      ('scoped_vmem_limit_kib', scoped_vmem_limit_kib),
      ('buffering_level', buffering_level),
      ('output_window_bounds', owb),
      ('kernel_window_bounds', kernel_window_bounds),
      ('input_window_bounds', input_window_bounds),
      ('integer', integer),
  ):
    if v is not None:
      base_cm[k] = _jax_k_fmt_meta(v)
  if isinstance(window_config, dict):
    for k, v in window_config.items():
      if v is not None:
        base_cm[str(k)] = _jax_k_fmt_meta(v)
  for k, v in extra.items():
    if v is not None:
      base_cm[k] = _jax_k_fmt_meta(v)
  if must_fuse and fusion_group is not None:
    base_cm['MUST_FUSE'] = _jax_k_fmt_meta(fusion_group)
  if isinstance(unsafe_metadata, dict):
    for k, v in unsafe_metadata.items():
      if v is not None:
        base_cm[str(k)] = _jax_k_fmt_meta(v)
  base_im: dict[str, Any] = dict({
      'MUST_FUSE': None,
      'MAXIMAL_FUSE': None,
      'FUSE_LIMIT': None,
      'scoped_vmem_limit_kib': None,
      'buffering_level': None,
      'output_window_bounds': None,
      'kernel_window_bounds': None,
      'input_window_bounds': None,
      'integer': None,
      'xla_disable_while_loop_copies': None,
      'xla_while_loop_async_pipelining_levels': None,
      'xla_pipeline_loop_carried_dus': None,
      'xla_while_loop_async_immediate_revisiting': None,
      'xla_post_fusion_pipelined_loop': None,
      'xla_disjoint_read_write_regions': None,
      'slice_sizes': None,
      'ColorToVMEM': None,
      'ColorInputToHBM': None,
      'explicit_layout': None,
  })
  for k in _clear_extra:
    base_im[k] = None
  if isinstance(unsafe_metadata, dict):
    for k in unsafe_metadata:
      base_im[str(k)] = None
  if isinstance(_flag_overrides, dict):
    base_im.update(_flag_overrides)

  def _dec(f: Any) -> Any:
    @functools.wraps(f)
    def _call(*a: Any, **kw: Any) -> Any:
      uid = _jax_k_next_fuid()
      bs = globals().get('_jax_k_bind_spill')
      if bs is not None:
        a, kw = jax.tree.map(bs, a), jax.tree.map(bs, kw)
      cm = dict(base_cm)
      if sv_imp is not None and not _has_ret_layout:
        v_ins = [
            x
            for idx, arg in enumerate(a)
            if idx not in _non_vmem_idxs
            for x in jax.tree.leaves(arg)
            if hasattr(x, 'shape')
        ] + [
            x
            for kn, kv in kw.items()
            if kn not in _non_vmem_names
            for x in jax.tree.leaves(kv)
            if hasattr(x, 'shape')
        ]
        if len(v_ins) < 2 and all(
            getattr(getattr(x, 'parent', None), 'primitive', None) is not None
            and getattr(getattr(x, 'parent', None), 'primitive').name
            == 'dynamic_slice'
            for x in v_ins
        ):
          try:
            if 'LayoutConstraint' not in str(jax.make_jaxpr(f)(*a, **kw)):
              cm['scoped_vmem_limit_kib'] = _jax_k_fmt_meta(sv_imp)
          except Exception:
            pass
      im = dict(base_im)
      if must_fuse:
        if fusion_group is None:
          im['MUST_FUSE'] = uid
      else:
        im['MAXIMAL_FUSE'] = (
            _jax_k_fmt_meta(fusion_group) if fusion_group is not None else uid
        )

      @functools.wraps(f)
      def _inner(*ia: Any, **ikw: Any) -> Any:
        with xla_metadata.set_xla_metadata(**im):
          return f(*ia, **ikw)

      return (
          jax_sched.xla_metadata_call(_inner, **cm)(*a, **kw)
          if cm
          else _inner(*a, **kw)
      )

    return _call

  return _dec if fn is None else _dec(fn)


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
      object.__setattr__(v, '_jax_k_cidx', idx)
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
    cd = getattr(c, '_jax_k_cidx', None)
    if isinstance(cd, int):
      return cd
    ce = _JAX_K_CARRY_IDS.get(id(c))
    if ce is not None and ce[0] is c:
      return ce[1]
    eq = getattr(c, 'parent', None)
    if eq is not None:
      for t in getattr(eq, 'in_tracers', ()):
        if getattr(t, 'ndim', None) == getattr(v, 'ndim', None):
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
    eq = getattr(c, 'parent', None)
    if eq is not None:
      ins = [
          t
          for t in getattr(eq, 'in_tracers', ())
          if getattr(t, 'ndim', None) == getattr(v, 'ndim', None)
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
    eq = getattr(c, 'parent', None)
    if eq is None:
      continue
    if getattr(eq.primitive, 'name', '') in (
        'add',
        'sub',
        'min',
        'max',
        'select_n',
        'clamp',
        'gather',
        'dynamic_slice',
    ):
      return True
    q.extend((t, d + 1) for t in getattr(eq, 'in_tracers', ()))
  return False


def _jax_k_rec_pad(
    t: Any, b_sz: tuple[int, ...], idx: Any, is_upd: bool
) -> None:
  if not _JAX_K_PAD_DISC or t is None or not hasattr(t, 'shape'):
    return
  ci = _jax_k_get_cidx(t)
  if ci is not None and ci in _JAX_K_CARRY_REQS:
    orig_s, reqs = _JAX_K_CARRY_REQS[ci]
  else:
    rt = _jax_k_get_src(t)
    if not hasattr(rt, 'shape'):
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
    for c in getattr(u, '__closure__', None) or ():
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
            'pipeline_loop_carried_dus',
            'immediate_revisiting',
            'async_pipelining_levels',
            'post_fusion_pipelined',
        )
    )
  b_m = dict(
      {
          'buffering_level': None,
          'xla_disable_while_loop_copies': None,
          'xla_while_loop_async_pipelining_levels': None,
          'xla_pipeline_loop_carried_dus': None,
          'xla_while_loop_async_immediate_revisiting': None,
          'xla_post_fusion_pipelined_loop': None,
      },
      **(
          {}
          if scoped_vmem_limit_kib is None
          else {'scoped_vmem_limit_kib': str(scoped_vmem_limit_kib)}
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
      (boundary_mode in (None, 'pad_hbm'))
      and _JAX_K_LOOP_DEPTH == 0
      and init is not None
      and jump_fn is None
  )
  glt, slt = globals().get('_jax_k_get_ltag', lambda _: None), globals().get(
      '_jax_k_set_ltag', lambda v, _: v
  )
  if run_disc:
    i_lvs, i_def = jax.tree.flatten(init)
    _JAX_K_CARRY_REQS.clear()
    _JAX_K_CLOS_REQS.clear()
    _JAX_K_CARRY_IDS.clear()
    _JAX_K_SRC_MAP.clear()
    for ci, lv in enumerate(i_lvs):
      _jax_k_set_cidx(lv, ci)
      if hasattr(lv, 'shape'):
        _JAX_K_CARRY_REQS[ci] = (
            tuple(int(d) for d in lv.shape),
            [int(d) for d in lv.shape],
        )
    for cl in _jax_k_collect_cells(body):
      try:
        for lv in jax.tree.leaves(cl.cell_contents):
          if hasattr(lv, 'shape'):
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
      os_t = getattr(lv, 'shape', ())
      en = _JAX_K_CARRY_REQS.get(ci)
      if en is not None and tuple(en[1]) != os_t:
        l_t = glt(lv)
        pl = jax.lax.pad(
            lv,
            jnp.zeros((), dtype=lv.dtype),
            tuple((0, max(0, r - d), 0) for d, r in zip(os_t, en[1])),
        )
        wl = globals().get('_jax_k_with_layout')
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
        'buffering_level': None,
        'xla_disable_while_loop_copies': None,
        'xla_while_loop_async_pipelining_levels': None,
        'xla_pipeline_loop_carried_dus': None,
        'xla_while_loop_async_immediate_revisiting': None,
        'xla_post_fusion_pipelined_loop': None,
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
      wl = globals().get('_jax_k_with_layout')
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
                if lt is not None and glt(nlv) is None and hasattr(nlv, 'shape')
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
  w_m = {'xla_disable_while_loop_copies': 'true'} if dc else {}
  if async_pipelining_levels is None and dc and (mb or not has_dus):
    async_pipelining_levels = buffering_level
  if pipeline_loop_carried_dus:
    w_m['xla_pipeline_loop_carried_dus'] = 'true'
  if post_fusion_pipelined and dc:
    w_m['xla_post_fusion_pipelined_loop'] = 'true'
  if immediate_revisiting is not None:
    w_m['xla_while_loop_async_immediate_revisiting'] = (
        'true' if immediate_revisiting else 'false'
    )
  if async_pipelining_levels is not None:
    w_m['xla_while_loop_async_pipelining_levels'] = str(async_pipelining_levels)
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


DEFAULT_MASK_VALUE = configs.DEFAULT_MASK_VALUE
LOG2E = configs.LOG2E
LOG2E_INV = configs.LOG2E_INV
LN2 = configs.LN2


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class SegmentIds:
  """Document boundary segment IDs for queries and keys/values."""

  q: jax.Array
  kv: jax.Array

  def tree_flatten(self):
    return ((self.q, self.kv), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


@dataclasses.dataclass(frozen=True)
class SplashBwdJaxKConfig:
  """Configuration and tunable tile sizes for jax_k Splash Attention backward.

  Attributes:
    block_q: Query sequence length tile size. Controls how many query tokens are
      loaded into vector memory per step.
    block_kv: Key and value sequence length tile size. Controls how many key and
      value tokens are processed in the outer loop.
    block_q_head: Number of query heads processed together per tile in GQA/MQA.
      If None, defaults to the full head group ratio (num_q_heads //
      num_kv_heads).
    block_kv_head: Number of key and value heads processed together per tile in
      GQA/MQA. If None, defaults to 1.
    mask_value: Additive value applied to masked attention positions before the
      softmax exponential (default -1e30).
    attn_logits_soft_cap: Optional threshold for attention logit soft capping:
      tanh(logits / cap) * cap. If None, no capping is applied.
    is_causal: If True, applies lower triangular causal masking (q_idx >=
      kv_idx).
    use_log2: If True, scales logits by log2(e) and uses base 2 exponentiation
      (2**(logits - lse)) for faster TPU evaluation.
    peel_diagonal: If True, separates the causal diagonal block (triangular
      mask) from off diagonal blocks (lower triangular blocks).
    mha_scoped_vmem_limit_kib: Scoped vector memory limit in KiB passed to
      Memory Space Assignment (MSA) for MHA to prevent memory spills.
    gqa_scoped_vmem_limit_kib: Scoped vector memory limit in KiB passed to
      Memory Space Assignment (MSA) for GQA.
  """

  block_q: int = 1024
  block_kv: int = 1024
  block_q_head: int | None = None
  block_kv_head: int | None = None
  mask_value: float = DEFAULT_MASK_VALUE
  attn_logits_soft_cap: float | None = None
  is_causal: bool = True
  use_log2: bool = True
  peel_diagonal: bool = True
  mha_scoped_vmem_limit_kib: int | None = 22528
  gqa_scoped_vmem_limit_kib: int | None = None


def _jax_k_loop(lower: Any, upper: Any, body_fn: Any, init_val: Any) -> Any:
  """Executes a non pipelined jax_k index loop lowering to stablehlo.while."""
  return _jax_k_pipeline(
      lower=lower,
      upper=upper,
      init=init_val,
      body=body_fn,
      disable_while_loop_copies=False,
      boundary_mode='peel',
      buffering_level=1,
  )


def _has_batch_tracer(tree: Any) -> Any:
  """Returns True if any leaf in `tree` is currently traced by `jax.vmap`."""
  return any(
      type(leaf).__name__ == 'BatchTracer'
      for leaf in jax.tree_util.tree_leaves(tree)
  )


def _is_staged_tracer(x: Any) -> Any:
  """Returns True if `x` is staged into a Jaxpr (such as inside `jax.jit`)."""
  while isinstance(x, jax.core.Tracer):
    if type(x).__name__ in ('DynamicJaxprTracer', 'JaxprTracer'):
      return True
    if hasattr(x, 'val'):
      x = x.val
    elif hasattr(x, 'primal'):
      x = x.primal
    else:
      return True
  return False


def _has_staged_tracer(tree: Any) -> Any:
  """Returns True if any leaf in `tree` is currently staged into a Jaxpr."""
  return any(
      _is_staged_tracer(leaf) for leaf in jax.tree_util.tree_leaves(tree)
  )


def _validate_mask_shape(mask: Any, q_shape: Any, kv_seq_len: int) -> Any:
  """Validates broadcast compatibility of an attention mask with `q` and `k`."""
  q_seq_len = q_shape[-2]
  num_q_heads = q_shape[-3]
  if mask.ndim < 2 or mask.ndim > len(q_shape):
    raise ValueError(
        f'mask must have rank between 2 and {len(q_shape)}, got {mask.shape}.'
    )
  if mask.shape[-2] != q_seq_len or mask.shape[-1] != kv_seq_len:
    raise ValueError(
        f'mask trailing dimensions {mask.shape[-2:]} must match'
        f' (q_seq_len={q_seq_len}, kv_seq_len={kv_seq_len}).'
    )
  if mask.ndim >= 3 and mask.shape[-3] not in (1, num_q_heads):
    raise ValueError(
        f'mask head dimension ({mask.shape[-3]}) must be 1 or num_q_heads'
        f' ({num_q_heads}), got {mask.shape}.'
    )
  mask_batch = mask.shape[:-3]
  q_batch = q_shape[len(q_shape) - 3 - len(mask_batch) : -3]
  for mb, qb in zip(mask_batch, q_batch):
    if mb not in (1, qb):
      raise ValueError(
          f'mask batch dimensions {mask_batch} are not broadcast compatible'
          f' with query batch dimensions {q_shape[:-3]}.'
      )


def _splash_attention_mha_bwd_jax_k_impl(
    q: Any,
    k: Any,
    v: Any,
    o: Any,
    do: Any,
    logsumexp: Any,
    segment_ids: Any,
    mask: Any,
    mask_value: Any,
    attn_logits_soft_cap: Any,
    block_q: int,
    block_kv: int,
    block_head: int,
    dtype: Any,
    is_causal: Any,
    use_log2: Any,
    di: Any = None,
    fuse_dus: Any = True,
    valid_q_len: Any = None,
    valid_kv_len: Any = None,
    q_offset: Any = None,
    q_prescaled_log2e: Any = False,
) -> Any:
  """jax_k implementation of Splash Attention backward pass for MHA.

  Computes query, key, and value gradients (dq, dk, dv) from forward tensors
  and upstream output gradients (do) for Multi Head Attention (MHA), where the
  number of query heads equals the number of key/value heads.

  Args:
    q: Query tensor of shape `(..., num_heads, q_seq_len, head_dim)`.
    k: Key tensor of shape `(..., num_heads, kv_seq_len, head_dim)`.
    v: Value tensor of shape `(..., num_heads, kv_seq_len, head_dim)`.
    o: Forward attention output of shape `(..., num_heads, q_seq_len,
      head_dim)`.
    do: Upstream output gradient of shape `(..., num_heads, q_seq_len,
      head_dim)`.
    logsumexp: Forward logsumexp normalizer of shape `(..., num_heads,
      q_seq_len)`.
    segment_ids: Optional document segment boundary IDs for masking.
    mask: Optional attention mask array.
    mask_value: Additive value applied to masked positions before
      exponentiation.
    attn_logits_soft_cap: Optional threshold for tanh logit soft capping.
    block_q: Sequence tile size for queries.
    block_kv: Sequence tile size for keys and values.
    block_head: Tile size along the head dimension.
    dtype: Computation precision dtype (float32 or bfloat16).
    is_causal: Whether lower triangular causal masking applies.
    use_log2: Whether base 2 exponentiation is used.
    di: Optional precomputed inner product sum(o * do, axis=-1).
    fuse_dus: Whether to fuse DynamicUpdateSlice into the sibling dot mega
      fusion. True for standalone benchmark calls, False when called under
      jax.vmap to avoid index clamping conflicts.
    valid_q_len: Optional unpadded query sequence length for boundary masking.
    valid_kv_len: Optional unpadded key/value sequence length for boundary
      masking.
    q_offset: Optional query position offset for causal masking.
    q_prescaled_log2e: Whether query tensor is already scaled by log2(e).

  Returns:
    A tuple of `(dq, dk, dv)` gradient arrays with the same shapes and
    precisions as the input `(q, k, v)` tensors.
  """
  q_seq_len = q.shape[-2]
  kv_seq_len = k.shape[-2]
  num_q_heads = q.shape[-3]
  if q_seq_len % block_q != 0:
    raise ValueError(
        f'q_seq_len ({q_seq_len}) must be divisible by block_q ({block_q}).'
    )
  if kv_seq_len % block_kv != 0:
    raise ValueError(
        f'kv_seq_len ({kv_seq_len}) must be divisible by block_kv ({block_kv}).'
    )
  if num_q_heads % block_head != 0:
    raise ValueError(
        f'num_q_heads ({num_q_heads}) must be divisible by block_head'
        f' ({block_head}).'
    )
  if mask is not None:
    _validate_mask_shape(mask, q.shape, kv_seq_len)
  if di is None:
    di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)
  else:
    di = di.astype(jnp.float32)
  logsumexp_prec = jnp.where(jnp.isneginf(logsumexp), 0.0, logsumexp).astype(
      jnp.float32
  )
  q_prec = q.astype(dtype)
  k_prec = k.astype(dtype)
  v_prec = v.astype(dtype)
  do_prec = do.astype(dtype)
  num_q_blocks = q_seq_len // block_q
  num_kv_blocks = kv_seq_len // block_kv
  has_q_offset = q_offset is not None
  peel_diagonal = (
      is_causal
      and block_q == block_kv
      and (q_seq_len == kv_seq_len)
      and (not has_q_offset)
  )
  apply_general_causal = bool(is_causal and (not peel_diagonal))
  has_seq_bounds = valid_q_len is not None or valid_kv_len is not None
  needs_tile_coords = apply_general_causal or has_seq_bounds
  eff_fuse_dus = bool(fuse_dus and block_q == block_kv and (block_q >= 512))
  eff_soft_cap = (
      attn_logits_soft_cap * LOG2E
      if attn_logits_soft_cap is not None and q_prescaled_log2e
      else attn_logits_soft_cap
  )
  mha_scoped_vmem_kib = 22528 if num_kv_blocks > 4 else None
  slice_sizes_k = k.shape[:-3] + (block_head, block_kv, k.shape[-1])
  slice_sizes_v = v.shape[:-3] + (block_head, block_kv, v.shape[-1])
  slice_sizes_q = q.shape[:-3] + (block_head, block_q, q.shape[-1])
  slice_sizes_do = do.shape[:-3] + (block_head, block_q, do.shape[-1])
  slice_sizes_log = logsumexp.shape[:-2] + (block_head, block_q)

  def slice_kv_block(start_idx_k, start_idx_v, unfused_dma: Any = False):
    """Loads a slice of key and value tensors into on chip vector memory."""
    if unfused_dma:
      with xla_metadata.set_xla_metadata(FUSE_LIMIT='true'):
        k_s = jax.lax.dynamic_slice(
            k_prec,
            tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_idx_k),
            slice_sizes_k,
        )
        v_s = jax.lax.dynamic_slice(
            v_prec,
            tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_idx_v),
            slice_sizes_v,
        )
    else:
      k_s = jax.lax.dynamic_slice(
          k_prec,
          tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_idx_k),
          slice_sizes_k,
      )
      v_s = jax.lax.dynamic_slice(
          v_prec,
          tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_idx_v),
          slice_sizes_v,
      )
    return (k_s, v_s)

  def slice_mask_tile(head_start_idx, q_start_idx, kv_start_idx):
    """Slices a tile from an optional 2D, 3D, or batched attention mask."""
    if mask is None:
      return None
    if mask.ndim == 2:
      return jax.lax.dynamic_slice(
          mask,
          (
              jnp.asarray(q_start_idx, dtype=jnp.int32),
              jnp.asarray(kv_start_idx, dtype=jnp.int32),
          ),
          (block_q, block_kv),
      )
    h_blk = 1 if mask.shape[-3] == 1 else block_head
    h_idx = 0 if mask.shape[-3] == 1 else head_start_idx
    return jax.lax.dynamic_slice(
        mask,
        tuple(
            jnp.asarray(i, dtype=jnp.int32)
            for i in (0,) * (mask.ndim - 3) + (h_idx, q_start_idx, kv_start_idx)
        ),
        mask.shape[:-3] + (h_blk, block_q, block_kv),
    )

  def eval_bwd_tile_fused(
      q_tile,
      k_tile,
      v_tile,
      do_tile,
      lse_tile,
      di_tile,
      mask_tile,
      q_seg_tile,
      kv_seg_tile,
      *,
      is_causal_diagonal: Any,
      dk_prev,
      dv_prev,
      dq_prev,
      accumulate_dq: Any,
      dq_carry_inner,
      q_slice_start,
      kv_slice_start,
      head_start,
      fuse_id: Any,
  ):
    """Evaluates the three output sibling dot mega fusion on one (Q, KV) tile."""
    if q_seg_tile is not None and kv_seg_tile is not None:
      seg_mask = (
          q_seg_tile[..., None, :, None] == kv_seg_tile[..., None, None, :]
      )
      if mask_tile is not None:
        mask_tile = jnp.logical_and(mask_tile, seg_mask)
      else:
        mask_tile = seg_mask
    if needs_tile_coords:
      q_local_1d = jnp.asarray(q_slice_start, dtype=jnp.int32) + jnp.arange(
          block_q, dtype=jnp.int32
      )
      q_causal_1d = (
          q_local_1d + jnp.asarray(q_offset, dtype=jnp.int32).reshape(())
          if q_offset is not None and apply_general_causal
          else q_local_1d
      )
      kv_local_1d = jnp.asarray(kv_slice_start, dtype=jnp.int32) + jnp.arange(
          block_kv, dtype=jnp.int32
      )
      coord_mask: Any = None
      if apply_general_causal:
        coord_mask = q_causal_1d[:, None] >= kv_local_1d[None, :]
      if valid_q_len is not None:
        q_valid = jnp.broadcast_to(
            (q_local_1d < valid_q_len)[:, None], (block_q, block_kv)
        )
        coord_mask = (
            q_valid
            if coord_mask is None
            else jnp.logical_and(coord_mask, q_valid)
        )
      if valid_kv_len is not None:
        kv_valid = jnp.broadcast_to(
            (kv_local_1d < valid_kv_len)[None, :], (block_q, block_kv)
        )
        coord_mask = (
            kv_valid
            if coord_mask is None
            else jnp.logical_and(coord_mask, kv_valid)
        )
      assert coord_mask is not None
      mask_tile = (
          coord_mask
          if mask_tile is None
          else jnp.logical_and(mask_tile, coord_mask)
      )
    has_mask = mask_tile is not None
    fuse_carries_inside = block_q == block_kv
    has_dk_prev = fuse_carries_inside and dk_prev is not None
    has_dv_prev = fuse_carries_inside and dv_prev is not None
    has_dq_prev = fuse_carries_inside and dq_prev is not None
    fuse_dus_inside = eff_fuse_dus and fuse_carries_inside

    @_jax_k_fusion(
        must_fuse=True,
        fusion_group=fuse_id,
        scoped_vmem_limit_kib=mha_scoped_vmem_kib,
    )
    def _fuse_bwd_tile(q_b, k_b, v_b, do_b, lse_b, di_b, *extra_args):
      idx = 0
      mask_b = extra_args[idx] if has_mask else None
      if has_mask:
        idx += 1
      dk_c = extra_args[idx] if has_dk_prev else None
      if has_dk_prev:
        idx += 1
      dv_c = extra_args[idx] if has_dv_prev else None
      if has_dv_prev:
        idx += 1
      dq_c = extra_args[idx] if has_dq_prev else None
      if has_dq_prev:
        idx += 1
      qk = jnp.einsum('...hd,...kd->...hk', q_b, k_b)
      qk_uncapped = qk
      if eff_soft_cap is not None:
        qk = jnp.tanh(qk_uncapped / eff_soft_cap) * eff_soft_cap
      qk_base = qk if q_prescaled_log2e else qk * LOG2E if use_log2 else qk
      if is_causal_diagonal:
        q_idx = lax.broadcasted_iota(jnp.int32, qk.shape, qk.ndim - 2)
        kv_idx = lax.broadcasted_iota(jnp.int32, qk.shape, qk.ndim - 1)
        qk_masked = jnp.where(q_idx >= kv_idx, qk_base, mask_value)
        if mask_b is not None:
          qk_masked = jnp.where(mask_b, qk_masked, mask_value)
      elif mask_b is not None:
        qk_masked = jnp.where(mask_b, qk_base, mask_value)
      else:
        qk_masked = qk_base
      lse_rhs = lse_b[..., None] if lse_b.ndim < qk_masked.ndim else lse_b
      di_rhs = di_b[..., None] if di_b.ndim < qk_masked.ndim else di_b
      diff_exp = (qk_masked.astype(jnp.float32) - lse_rhs).astype(q_b.dtype)
      if use_log2:
        p = 2**diff_exp
      else:
        p = jnp.exp(diff_exp)
      dv_upd = jnp.einsum('...hk,...hd->...kd', p, do_b)
      if dv_c is not None:
        dv_upd = dv_upd + dv_c
      dp = jnp.einsum('...hd,...kd->...hk', do_b, v_b)
      dp_minus_di = (dp.astype(jnp.float32) - di_rhs).astype(q_b.dtype)
      ds = p * dp_minus_di
      if eff_soft_cap is not None:
        normalized = qk_uncapped / eff_soft_cap
        d = jnp.tanh(normalized)
        g = ds * (1 - d)
        ds = (g + g * d).astype(q_b.dtype)
      dk_upd = jnp.einsum('...hk,...hd->...kd', ds, q_b)
      if dk_c is not None:
        dk_upd = dk_upd + dk_c
      dq_upd = jnp.einsum('...hk,...kd->...hd', ds, k_b)
      if dq_c is not None:
        dq_upd = dq_upd + dq_c
      if fuse_dus_inside:
        dq_full_c = extra_args[idx]
        q_start_c = extra_args[idx + 1]
        h_start_c = extra_args[idx + 2]
        start_indices_dus = (0,) * (dq_full_c.ndim - 3) + (
            h_start_c,
            q_start_c,
            0,
        )
        if dq_c is None and accumulate_dq:
          dq_upd = dq_upd + jax.lax.dynamic_slice(
              dq_full_c,
              tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_dus),
              dq_upd.shape,
          )
        dq_out = lax.dynamic_update_slice(dq_full_c, dq_upd, start_indices_dus)
        return (dk_upd, dv_upd, dq_out)
      return (dk_upd, dv_upd, dq_upd)

    extra_inputs = []
    if has_mask:
      extra_inputs.append(mask_tile)
    if has_dk_prev:
      extra_inputs.append(dk_prev)
    if has_dv_prev:
      extra_inputs.append(dv_prev)
    if has_dq_prev:
      extra_inputs.append(dq_prev)
    if fuse_dus_inside:
      extra_inputs.append(dq_carry_inner)
      extra_inputs.append(jnp.asarray(q_slice_start, dtype=jnp.int32))
      extra_inputs.append(jnp.asarray(head_start, dtype=jnp.int32))
    res_fused = _fuse_bwd_tile(
        q_tile, k_tile, v_tile, do_tile, lse_tile, di_tile, *extra_inputs
    )
    if fuse_dus_inside:
      return res_fused
    dk_upd, dv_upd, dq_upd = res_fused
    if not fuse_carries_inside:
      if dk_prev is not None:
        dk_upd = dk_upd + dk_prev
      if dv_prev is not None:
        dv_upd = dv_upd + dv_prev
      if dq_prev is not None:
        dq_upd = dq_upd + dq_prev
    start_indices_dus = (0,) * (dq_carry_inner.ndim - 3) + (
        head_start,
        q_slice_start,
        0,
    )
    if dq_prev is None and accumulate_dq:
      dq_upd = dq_upd + jax.lax.dynamic_slice(
          dq_carry_inner,
          tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_dus),
          dq_upd.shape,
      )
    dq_out = lax.dynamic_update_slice(dq_carry_inner, dq_upd, start_indices_dus)
    return (dk_upd, dv_upd, dq_out)

  def head_kernel(head_idx, head_carry):
    """Processes a chunk of attention heads."""
    head_start = head_idx * block_head

    def k_block_kernel(i, carry):
      """Processes one key/value block across all valid query sequence blocks."""
      dq_carry, dk_carry, dv_carry = carry
      kv_slice_start = i * block_kv
      start_indices_k = (0,) * (k.ndim - 3) + (head_start, kv_slice_start, 0)
      start_indices_v = (0,) * (v.ndim - 3) + (head_start, kv_slice_start, 0)
      shared_kv = (
          slice_kv_block(start_indices_k, start_indices_v, unfused_dma=True)
          if num_kv_blocks > 4
          else None
      )
      kv_segment_ids_block = (
          lax.dynamic_slice_in_dim(
              segment_ids.kv, kv_slice_start, block_kv, axis=-1
          )
          if segment_ids is not None
          else None
      )
      if peel_diagonal:
        start_indices_diag_q = (0,) * (q.ndim - 3) + (
            head_start,
            kv_slice_start,
            0,
        )
        start_indices_diag_log = (0,) * (logsumexp.ndim - 2) + (
            head_start,
            kv_slice_start,
        )
        q_diag = jax.lax.dynamic_slice(
            q_prec,
            tuple(
                jnp.asarray(i, dtype=jnp.int32) for i in start_indices_diag_q
            ),
            slice_sizes_q,
        )
        k_diag, v_diag = (
            shared_kv
            if shared_kv is not None
            else slice_kv_block(start_indices_k, start_indices_v)
        )
        do_diag = jax.lax.dynamic_slice(
            do_prec,
            tuple(
                jnp.asarray(i, dtype=jnp.int32) for i in start_indices_diag_q
            ),
            slice_sizes_do,
        )
        lse_diag = jax.lax.dynamic_slice(
            logsumexp_prec,
            tuple(
                jnp.asarray(i, dtype=jnp.int32) for i in start_indices_diag_log
            ),
            slice_sizes_log,
        )
        di_diag = jax.lax.dynamic_slice(
            di,
            tuple(
                jnp.asarray(i, dtype=jnp.int32) for i in start_indices_diag_log
            ),
            slice_sizes_log,
        )
        dq_diag_slice = jax.lax.dynamic_slice(
            dq_carry,
            tuple(
                jnp.asarray(i, dtype=jnp.int32) for i in start_indices_diag_q
            ),
            slice_sizes_q,
        )
        mask_diag = slice_mask_tile(head_start, kv_slice_start, kv_slice_start)
        q_diag_seg = (
            lax.dynamic_slice_in_dim(
                segment_ids.q, kv_slice_start, block_q, axis=-1
            )
            if segment_ids is not None
            else None
        )
        dk_block_init, dv_block_init, dq_carry = eval_bwd_tile_fused(
            q_diag,
            k_diag,
            v_diag,
            do_diag,
            lse_diag,
            di_diag,
            mask_diag,
            q_diag_seg,
            kv_segment_ids_block,
            is_causal_diagonal=True,
            dk_prev=None,
            dv_prev=None,
            dq_prev=dq_diag_slice,
            accumulate_dq=False,
            dq_carry_inner=dq_carry,
            q_slice_start=kv_slice_start,
            kv_slice_start=kv_slice_start,
            head_start=head_start,
            fuse_id='11',
        )
        start_j = i + 1
      else:
        dk_block_init = jnp.zeros(slice_sizes_k, dtype=dtype)
        dv_block_init = jnp.zeros(slice_sizes_v, dtype=dtype)
        start_j = 0

      def q_block_kernel(j, inner_carry):
        """Processes one query block against the current key and value block."""
        dk_block_carry, dv_block_carry, dq_carry_inner = inner_carry
        q_slice_start = j * block_q
        start_indices_q = (0,) * (q.ndim - 3) + (head_start, q_slice_start, 0)
        start_indices_do = (0,) * (do.ndim - 3) + (head_start, q_slice_start, 0)
        start_indices_log = (0,) * (logsumexp.ndim - 2) + (
            head_start,
            q_slice_start,
        )
        q_block = jax.lax.dynamic_slice(
            q_prec,
            tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_q),
            slice_sizes_q,
        )
        k_block, v_block = (
            shared_kv
            if shared_kv is not None
            else slice_kv_block(start_indices_k, start_indices_v)
        )
        do_block = jax.lax.dynamic_slice(
            do_prec,
            tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_do),
            slice_sizes_do,
        )
        lse_block = jax.lax.dynamic_slice(
            logsumexp_prec,
            tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_log),
            slice_sizes_log,
        ).reshape(slice_sizes_log + (1,))
        di_block = jax.lax.dynamic_slice(
            di,
            tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_log),
            slice_sizes_log,
        ).reshape(slice_sizes_log + (1,))
        dq_slice = jax.lax.dynamic_slice(
            dq_carry_inner,
            tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_q),
            slice_sizes_q,
        )
        q_segment_ids_block = (
            lax.dynamic_slice_in_dim(
                segment_ids.q, q_slice_start, block_q, axis=-1
            )
            if segment_ids is not None
            else None
        )
        mask_block_qkv = slice_mask_tile(
            head_start, q_slice_start, kv_slice_start
        )
        return eval_bwd_tile_fused(
            q_block,
            k_block,
            v_block,
            do_block,
            lse_block,
            di_block,
            mask_block_qkv,
            q_segment_ids_block,
            kv_segment_ids_block,
            is_causal_diagonal=False,
            dk_prev=dk_block_carry,
            dv_prev=dv_block_carry,
            dq_prev=dq_slice,
            accumulate_dq=False,
            dq_carry_inner=dq_carry_inner,
            q_slice_start=q_slice_start,
            kv_slice_start=kv_slice_start,
            head_start=head_start,
            fuse_id='12',
        )

      if num_q_blocks > 1 or not peel_diagonal:
        dk_block, dv_block, dq_carry = _jax_k_loop(
            start_j,
            num_q_blocks,
            q_block_kernel,
            (dk_block_init, dv_block_init, dq_carry),
        )
      else:
        dk_block, dv_block = (dk_block_init, dv_block_init)
      dk_carry = lax.dynamic_update_slice(dk_carry, dk_block, start_indices_k)
      dv_carry = lax.dynamic_update_slice(dv_carry, dv_block, start_indices_v)
      return (dq_carry, dk_carry, dv_carry)

    if num_kv_blocks == 1:
      return k_block_kernel(0, head_carry)
    return _jax_k_loop(0, num_kv_blocks, k_block_kernel, head_carry)

  dk = lax.empty(k.shape, dtype=dtype)
  dv = lax.empty(v.shape, dtype=dtype)
  dq = jnp.zeros(q.shape, dtype=dtype)
  num_head_blocks = num_q_heads // block_head
  if num_head_blocks == 1:
    dq, dk, dv = head_kernel(0, (dq, dk, dv))
  else:
    dq, dk, dv = _jax_k_loop(0, num_head_blocks, head_kernel, (dq, dk, dv))
  return (dq, dk, dv)


_splash_attention_mha_bwd_jax_k_jit = jax.jit(
    _splash_attention_mha_bwd_jax_k_impl,
    static_argnames=[
        'block_q',
        'block_kv',
        'block_head',
        'dtype',
        'is_causal',
        'use_log2',
        'fuse_dus',
        'valid_q_len',
        'valid_kv_len',
        'q_prescaled_log2e',
    ],
)


def splash_attention_mha_bwd_jax_k(
    q: Any,
    k: Any,
    v: Any,
    o: Any,
    do: Any,
    logsumexp: Any,
    segment_ids: Any,
    mask: Any,
    mask_value: Any,
    attn_logits_soft_cap: Any,
    block_q: int,
    block_kv: int,
    block_head: int,
    dtype: Any,
    is_causal: Any,
    use_log2: Any,
    di: Any = None,
    fuse_dus: Any = True,
    valid_q_len: Any = None,
    valid_kv_len: Any = None,
    q_offset: Any = None,
    q_prescaled_log2e: Any = False,
) -> Any:
  """Dispatches jax_k MHA backward pass, disabling fused DUS under `jax.vmap`."""
  array_args = (q, k, v, o, do, logsumexp, segment_ids, mask, di, q_offset)

  def _invoke_mha(
      q_a,
      k_a,
      v_a,
      o_a,
      do_a,
      lse_a,
      seg_a,
      mask_a,
      di_a,
      qoff_a,
      dus_flag: Any,
  ):
    staged = _has_staged_tracer((
        q_a,
        k_a,
        v_a,
        o_a,
        do_a,
        lse_a,
        seg_a,
        mask_a,
        di_a,
        qoff_a,
        mask_value,
        attn_logits_soft_cap,
    ))
    fn = (
        _splash_attention_mha_bwd_jax_k_impl
        if staged
        else _splash_attention_mha_bwd_jax_k_jit
    )
    return fn(
        q_a,
        k_a,
        v_a,
        o_a,
        do_a,
        lse_a,
        seg_a,
        mask_a,
        mask_value,
        attn_logits_soft_cap,
        block_q,
        block_kv,
        block_head,
        dtype,
        is_causal,
        use_log2,
        di_a,
        dus_flag,
        valid_q_len,
        valid_kv_len,
        qoff_a,
        q_prescaled_log2e,
    )

  if not fuse_dus or _has_batch_tracer(
      (array_args, mask_value, attn_logits_soft_cap)
  ):
    return _invoke_mha(
        q, k, v, o, do, logsumexp, segment_ids, mask, di, q_offset, False
    )

  def _run_mha(dus_flag: Any, args_tuple: Any):
    q_a, k_a, v_a, o_a, do_a, lse_a, seg_a, mask_a, di_a, qoff_a = args_tuple
    return _invoke_mha(
        q_a, k_a, v_a, o_a, do_a, lse_a, seg_a, mask_a, di_a, qoff_a, dus_flag
    )

  @jax.custom_batching.custom_vmap
  def _call_mha(args_tuple):
    return _run_mha(True, args_tuple)

  @_call_mha.def_vmap
  def _call_mha_vmap(axis_size, in_batched, args_tuple):
    del axis_size
    if not any(jax.tree_util.tree_leaves(in_batched)):
      return (_run_mha(False, args_tuple), (False, False, False))
    in_axes = (
        jax.tree_util.tree_map(lambda b: 0 if b else None, in_batched[0]),
    )
    out = jax.vmap(lambda a: _run_mha(False, a), in_axes=in_axes)(args_tuple)
    return (out, (True, True, True))

  return _call_mha(array_args)


def _splash_attention_gqa_bwd_jax_k_impl(
    q: Any,
    k: Any,
    v: Any,
    o: Any,
    do: Any,
    logsumexp: Any,
    segment_ids: Any,
    mask: Any,
    block_q: int,
    block_kv: int,
    block_q_head: int,
    block_kv_head: int,
    mask_value: Any,
    attn_logits_soft_cap: Any = None,
    dtype: Any = jnp.float32,
    is_causal: Any = False,
    use_log2: Any = False,
    di: Any = None,
    fuse_dus: Any = True,
    valid_q_len: Any = None,
    valid_kv_len: Any = None,
    q_offset: Any = None,
    q_prescaled_log2e: Any = False,
) -> Any:
  """jax_k implementation of Splash Attention backward pass for GQA and MQA.

  Computes query, key, and value gradients (dq, dk, dv) for Grouped Query
  Attention (GQA) and Multi Query Attention (MQA), where multiple query heads
  share each key and value head.

  In GQA, the key and value gradients accumulate across both the sequence
  tokens and all query heads belonging to the respective KV head group:
    dV_kv = sum_{q in group(kv)} P_q.T @ dO_q
    dK_kv = sum_{q in group(kv)} dS_q.T @ Q_q
    dQ_q  = dS_q @ K_kv

  Args:
    q: Query tensor of shape `(..., num_q_heads, q_seq_len, head_dim)`.
    k: Key tensor of shape `(..., num_kv_heads, kv_seq_len, head_dim)`.
    v: Value tensor of shape `(..., num_kv_heads, kv_seq_len, head_dim)`.
    o: Forward attention output of shape `(..., num_q_heads, q_seq_len,
      head_dim)`.
    do: Upstream output gradient of shape `(..., num_q_heads, q_seq_len,
      head_dim)`.
    logsumexp: Forward logsumexp normalizer of shape `(..., num_q_heads,
      q_seq_len)`.
    segment_ids: Optional document segment boundary IDs for masking.
    mask: Optional attention mask array.
    block_q: Sequence tile size for queries.
    block_kv: Sequence tile size for keys and values.
    block_q_head: Number of query heads per group evaluated in a single tile.
    block_kv_head: Number of key and value heads processed per tile.
    mask_value: Additive value applied to masked positions before
      exponentiation.
    attn_logits_soft_cap: Optional threshold for tanh logit soft capping.
    dtype: Computation precision dtype (float32 or bfloat16).
    is_causal: Whether lower triangular causal masking applies.
    use_log2: Whether base 2 exponentiation is used.
    di: Optional precomputed inner product sum(o * do, axis=-1).
    fuse_dus: Whether to fuse DynamicUpdateSlice into the sibling dot mega
      fusion. True for standalone benchmark calls, False when called under
      jax.vmap to avoid index clamping conflicts.
    valid_q_len: Optional unpadded query sequence length for boundary masking.
    valid_kv_len: Optional unpadded key/value sequence length for boundary
      masking.
    q_offset: Optional query position offset for causal masking.
    q_prescaled_log2e: Whether query tensor is already scaled by log2(e).

  Returns:
    A tuple of `(dq, dk, dv)` gradient arrays with the same shapes and
    precisions as the input `(q, k, v)` tensors.
  """
  q_seq_len = q.shape[-2]
  kv_seq_len = k.shape[-2]
  num_q_heads = q.shape[-3]
  num_kv_heads = k.shape[-3]
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f'num_q_heads ({num_q_heads}) must be divisible by num_kv_heads'
        f' ({num_kv_heads}).'
    )
  head_multiplier = num_q_heads // num_kv_heads
  if q_seq_len % block_q != 0:
    raise ValueError(
        f'q_seq_len ({q_seq_len}) must be divisible by block_q ({block_q}).'
    )
  if kv_seq_len % block_kv != 0:
    raise ValueError(
        f'kv_seq_len ({kv_seq_len}) must be divisible by block_kv ({block_kv}).'
    )
  if num_kv_heads % block_kv_head != 0:
    raise ValueError(
        f'num_kv_heads ({num_kv_heads}) must be divisible by block_kv_head'
        f' ({block_kv_head}).'
    )
  if head_multiplier % block_q_head != 0:
    raise ValueError(
        f'head_multiplier ({head_multiplier}) must be divisible by block_q_head'
        f' ({block_q_head}).'
    )
  num_kv_head_blocks = num_kv_heads // block_kv_head
  num_q_head_blocks = head_multiplier // block_q_head
  if mask is not None:
    _validate_mask_shape(mask, q.shape, kv_seq_len)
  if di is None:
    di = jnp.sum(o * do, axis=-1, dtype=dtype)
  else:
    di = di.astype(dtype)
  logsumexp_prec = jnp.where(jnp.isneginf(logsumexp), 0.0, logsumexp).astype(
      dtype
  )
  q_prec = q.astype(dtype)
  k_prec = k.astype(dtype)
  v_prec = v.astype(dtype)
  do_prec = do.astype(dtype)
  multi_kv_head_tile = block_kv_head > 1
  batch_dims = q.shape[:-3]
  if multi_kv_head_tile:
    q_work = q_prec.reshape(
        batch_dims + (num_kv_heads, head_multiplier, q_seq_len, q.shape[-1])
    )
    do_work = do_prec.reshape(
        batch_dims + (num_kv_heads, head_multiplier, q_seq_len, do.shape[-1])
    )
    lse_work = logsumexp_prec.reshape(
        batch_dims + (num_kv_heads, head_multiplier, q_seq_len)
    )
    di_work = di.reshape(
        batch_dims + (num_kv_heads, head_multiplier, q_seq_len)
    )
    if mask is not None and mask.ndim >= 3:
      if mask.shape[-3] == 1:
        mask_work = mask.reshape(
            mask.shape[:-3] + (1, 1, q_seq_len, kv_seq_len)
        )
      else:
        mask_work = mask.reshape(
            mask.shape[:-3]
            + (num_kv_heads, head_multiplier, q_seq_len, kv_seq_len)
        )
    else:
      mask_work = mask
  else:
    q_work = q_prec
    do_work = do_prec
    lse_work = logsumexp_prec
    di_work = di
    mask_work = mask
  num_q_blocks = q_seq_len // block_q
  num_kv_blocks = kv_seq_len // block_kv
  has_q_offset = q_offset is not None
  peel_diagonal = (
      is_causal
      and block_q == block_kv
      and (q_seq_len == kv_seq_len)
      and (not has_q_offset)
  )
  apply_general_causal = bool(is_causal and (not peel_diagonal))
  has_seq_bounds = valid_q_len is not None or valid_kv_len is not None
  needs_tile_coords = apply_general_causal or has_seq_bounds
  eff_fuse_dus = bool(
      fuse_dus
      and block_q == block_kv
      and (block_q >= 512)
      and (not multi_kv_head_tile)
  )
  k_prescaled = bool(
      use_log2
      and (not q_prescaled_log2e)
      and (attn_logits_soft_cap is None)
      and (jnp.dtype(dtype) == jnp.bfloat16)
  )
  eff_soft_cap = (
      attn_logits_soft_cap * LOG2E
      if attn_logits_soft_cap is not None and q_prescaled_log2e
      else attn_logits_soft_cap
  )
  gqa_scoped_vmem_kib = None
  slice_sizes_k = batch_dims + (block_kv_head, block_kv, k.shape[-1])
  slice_sizes_v = batch_dims + (block_kv_head, block_kv, v.shape[-1])
  if multi_kv_head_tile:
    q_head_tile_dims: tuple[int, ...] = (block_kv_head, block_q_head)
  else:
    q_head_tile_dims = (block_q_head,)
  slice_sizes_q = batch_dims + q_head_tile_dims + (block_q, q.shape[-1])
  slice_sizes_do = batch_dims + q_head_tile_dims + (block_q, do.shape[-1])
  slice_sizes_log = batch_dims + q_head_tile_dims + (block_q,)

  def _broadcast_kv_to_q_heads(x: Any) -> Any:
    if multi_kv_head_tile:
      return jnp.broadcast_to(
          x[..., :, None, :, :],
          batch_dims + (block_kv_head, block_q_head, block_kv, x.shape[-1]),
      )
    return jnp.broadcast_to(
        x, batch_dims + (block_q_head, block_kv, x.shape[-1])
    )

  def slice_kv_block(start_idx_k, start_idx_v, unfused_dma: Any = False):
    """Loads a slice of K and V, pre scales K in outer loop, and broadcasts."""
    if unfused_dma:
      with xla_metadata.set_xla_metadata(FUSE_LIMIT='true'):
        k_s = jax.lax.dynamic_slice(
            k_prec,
            tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_idx_k),
            slice_sizes_k,
        )
        v_s = jax.lax.dynamic_slice(
            v_prec,
            tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_idx_v),
            slice_sizes_v,
        )
    else:
      k_s = jax.lax.dynamic_slice(
          k_prec,
          tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_idx_k),
          slice_sizes_k,
      )
      v_s = jax.lax.dynamic_slice(
          v_prec,
          tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_idx_v),
          slice_sizes_v,
      )
    k_s_scaled = (
        (k_s.astype(jnp.float32) * LOG2E).astype(k_s.dtype)
        if k_prescaled
        else None
    )
    k_bcast = _broadcast_kv_to_q_heads(k_s)
    v_bcast = _broadcast_kv_to_q_heads(v_s)
    k_scaled_bcast = (
        _broadcast_kv_to_q_heads(k_s_scaled) if k_s_scaled is not None else None
    )
    return (k_bcast, v_bcast, k_scaled_bcast)

  def slice_mask_tile(
      kv_head_start_idx, q_sub_head_start_idx, q_head_flat_idx, q_idx, kv_idx
  ):
    """Slices a tile from an optional 2D, 3D, or batched attention mask."""
    if mask_work is None:
      return None
    if mask_work.ndim == 2:
      return jax.lax.dynamic_slice(
          mask_work,
          (
              jnp.asarray(q_idx, dtype=jnp.int32),
              jnp.asarray(kv_idx, dtype=jnp.int32),
          ),
          (block_q, block_kv),
      )
    if multi_kv_head_tile:
      kv_h_blk = 1 if mask_work.shape[-4] == 1 else block_kv_head
      kv_h_idx = 0 if mask_work.shape[-4] == 1 else kv_head_start_idx
      q_h_blk = 1 if mask_work.shape[-3] == 1 else block_q_head
      q_h_idx = 0 if mask_work.shape[-3] == 1 else q_sub_head_start_idx
      return jax.lax.dynamic_slice(
          mask_work,
          tuple(
              jnp.asarray(i, dtype=jnp.int32)
              for i in (0,) * (mask_work.ndim - 4) + (
                  kv_h_idx,
                  q_h_idx,
                  q_idx,
                  kv_idx,
              )
          ),
          mask_work.shape[:-4] + (kv_h_blk, q_h_blk, block_q, block_kv),
      )
    q_h_blk = 1 if mask_work.shape[-3] == 1 else block_q_head
    q_h_idx = 0 if mask_work.shape[-3] == 1 else q_head_flat_idx
    return jax.lax.dynamic_slice(
        mask_work,
        tuple(
            jnp.asarray(i, dtype=jnp.int32)
            for i in (0,) * (mask_work.ndim - 3) + (q_h_idx, q_idx, kv_idx)
        ),
        mask_work.shape[:-3] + (q_h_blk, block_q, block_kv),
    )

  def eval_bwd_tile_fused(
      q_tile,
      k_tile,
      v_tile,
      do_tile,
      lse_tile,
      di_tile,
      mask_tile,
      q_seg_tile,
      kv_seg_tile,
      *,
      is_causal_diagonal: Any,
      dk_prev,
      dv_prev,
      dq_prev,
      accumulate_dq: Any,
      dq_carry_inner,
      q_slice_start,
      kv_slice_start,
      q_head_indices,
      fuse_id: Any,
      k_scaled_tile=None,
  ):
    """Evaluates the three output sibling dot mega fusion for one GQA tile."""
    if q_seg_tile is not None and kv_seg_tile is not None:
      if multi_kv_head_tile:
        seg_mask = (
            q_seg_tile[..., None, None, :, None]
            == kv_seg_tile[..., None, None, None, :]
        )
      else:
        seg_mask = (
            q_seg_tile[..., None, :, None] == kv_seg_tile[..., None, None, :]
        )
      if mask_tile is not None:
        mask_tile = jnp.logical_and(mask_tile, seg_mask)
      else:
        mask_tile = seg_mask
    if needs_tile_coords:
      q_local_1d = jnp.asarray(q_slice_start, dtype=jnp.int32) + jnp.arange(
          block_q, dtype=jnp.int32
      )
      q_causal_1d = (
          q_local_1d + jnp.asarray(q_offset, dtype=jnp.int32).reshape(())
          if q_offset is not None and apply_general_causal
          else q_local_1d
      )
      kv_local_1d = jnp.asarray(kv_slice_start, dtype=jnp.int32) + jnp.arange(
          block_kv, dtype=jnp.int32
      )
      coord_mask: Any = None
      if apply_general_causal:
        coord_mask = q_causal_1d[:, None] >= kv_local_1d[None, :]
      if valid_q_len is not None:
        q_valid = jnp.broadcast_to(
            (q_local_1d < valid_q_len)[:, None], (block_q, block_kv)
        )
        coord_mask = (
            q_valid
            if coord_mask is None
            else jnp.logical_and(coord_mask, q_valid)
        )
      if valid_kv_len is not None:
        kv_valid = jnp.broadcast_to(
            (kv_local_1d < valid_kv_len)[None, :], (block_q, block_kv)
        )
        coord_mask = (
            kv_valid
            if coord_mask is None
            else jnp.logical_and(coord_mask, kv_valid)
        )
      assert coord_mask is not None
      mask_tile = (
          coord_mask
          if mask_tile is None
          else jnp.logical_and(mask_tile, coord_mask)
      )
    has_mask = mask_tile is not None
    fuse_carries_inside = block_q == block_kv
    has_dk_prev = fuse_carries_inside and dk_prev is not None
    has_dv_prev = fuse_carries_inside and dv_prev is not None
    has_dq_prev = fuse_carries_inside and dq_prev is not None
    fuse_dus_inside = eff_fuse_dus and fuse_carries_inside

    @_jax_k_fusion(
        must_fuse=True,
        fusion_group=fuse_id,
        scoped_vmem_limit_kib=gqa_scoped_vmem_kib,
    )
    def _fuse_bwd_tile(q_b, k_b, v_b, do_b, lse_b, di_b, *extra_args):
      idx = 0
      if k_prescaled:
        k_for_qk = extra_args[idx]
        idx += 1
      else:
        k_for_qk = k_b
      mask_b = extra_args[idx] if has_mask else None
      if has_mask:
        idx += 1
      dk_c = extra_args[idx] if has_dk_prev else None
      if has_dk_prev:
        idx += 1
      dv_c = extra_args[idx] if has_dv_prev else None
      if has_dv_prev:
        idx += 1
      dq_c = extra_args[idx] if has_dq_prev else None
      if has_dq_prev:
        idx += 1
      qk = jnp.einsum(
          '...hd,...kd->...hk',
          q_b,
          k_for_qk,
          preferred_element_type=jnp.float32,
      )
      qk_uncapped = qk
      if eff_soft_cap is not None:
        qk = jnp.tanh(qk_uncapped / eff_soft_cap) * eff_soft_cap
      qk_base = (
          qk
          if k_prescaled or q_prescaled_log2e
          else qk * LOG2E
          if use_log2
          else qk
      )
      if is_causal_diagonal:
        q_idx = lax.broadcasted_iota(jnp.int32, qk.shape, qk.ndim - 2)
        kv_idx = lax.broadcasted_iota(jnp.int32, qk.shape, qk.ndim - 1)
        qk_masked = jnp.where(q_idx >= kv_idx, qk_base, mask_value)
        if mask_b is not None:
          qk_masked = jnp.where(mask_b, qk_masked, mask_value)
      elif mask_b is not None:
        qk_masked = jnp.where(mask_b, qk_base, mask_value)
      else:
        qk_masked = qk_base
      if lse_b.ndim < qk_masked.ndim:
        lse_b = lse_b[..., None].astype(jnp.float32)
      if di_b.ndim < qk_masked.ndim:
        di_b = di_b[..., None].astype(jnp.float32)
      if use_log2:
        p = 2 ** (qk_masked - lse_b)
      else:
        p = jnp.exp(qk_masked - lse_b)
      p = p.astype(q_b.dtype)
      dv_upd = jnp.einsum('...hk,...hd->...kd', p, do_b)
      if dv_c is not None:
        dv_upd = dv_upd + dv_c
      dp = jnp.einsum(
          '...hd,...kd->...hk', do_b, v_b, preferred_element_type=jnp.float32
      )
      ds = p.astype(jnp.float32) * (dp - di_b)
      if eff_soft_cap is not None:
        normalized = qk_uncapped / eff_soft_cap
        d = jnp.tanh(normalized)
        g = ds * (1 - d)
        ds = g + g * d
      ds = ds.astype(q_b.dtype)
      dk_upd = jnp.einsum('...hk,...hd->...kd', ds, q_b)
      if dk_c is not None:
        dk_upd = dk_upd + dk_c
      dq_upd = jnp.einsum('...hk,...kd->...hd', ds, k_b)
      if dq_c is not None:
        dq_upd = dq_upd + dq_c
      if fuse_dus_inside:
        dq_full_c = extra_args[idx]
        q_start_c = extra_args[idx + 1]
        h_start_c = extra_args[idx + 2]
        start_indices_dus = (0,) * (dq_full_c.ndim - 3) + (
            h_start_c,
            q_start_c,
            0,
        )
        if dq_c is None and accumulate_dq:
          dq_upd = dq_upd + jax.lax.dynamic_slice(
              dq_full_c,
              tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_dus),
              dq_upd.shape,
          )
        dq_out = lax.dynamic_update_slice(dq_full_c, dq_upd, start_indices_dus)
        return (dk_upd, dv_upd, dq_out)
      return (dk_upd, dv_upd, dq_upd)

    extra_inputs = []
    if k_prescaled:
      extra_inputs.append(k_scaled_tile)
    if has_mask:
      extra_inputs.append(mask_tile)
    if has_dk_prev:
      extra_inputs.append(dk_prev)
    if has_dv_prev:
      extra_inputs.append(dv_prev)
    if has_dq_prev:
      extra_inputs.append(dq_prev)
    if fuse_dus_inside:
      extra_inputs.append(dq_carry_inner)
      extra_inputs.append(jnp.asarray(q_slice_start, dtype=jnp.int32))
      extra_inputs.append(jnp.asarray(q_head_indices[0], dtype=jnp.int32))
    res_fused = _fuse_bwd_tile(
        q_tile, k_tile, v_tile, do_tile, lse_tile, di_tile, *extra_inputs
    )
    if fuse_dus_inside:
      return res_fused
    dk_upd, dv_upd, dq_upd = res_fused
    if not fuse_carries_inside:
      if dk_prev is not None:
        dk_upd = dk_upd + dk_prev
      if dv_prev is not None:
        dv_upd = dv_upd + dv_prev
      if dq_prev is not None:
        dq_upd = dq_upd + dq_prev
    start_indices_dus = (
        (0,) * len(batch_dims) + q_head_indices + (q_slice_start, 0)
    )
    if dq_prev is None and accumulate_dq:
      dq_upd = dq_upd + jax.lax.dynamic_slice(
          dq_carry_inner,
          tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_dus),
          dq_upd.shape,
      )
    dq_out = lax.dynamic_update_slice(dq_carry_inner, dq_upd, start_indices_dus)
    return (dk_upd, dv_upd, dq_out)

  def kv_head_kernel(kv_head_idx, kv_head_carry):
    """Processes one chunk of key and value heads."""
    dk_kv, dv_kv, dq_kv = kv_head_carry
    kv_head_start = kv_head_idx * block_kv_head

    def q_head_kernel(q_head_idx, q_head_carry):
      """Processes query heads belonging to the current key/value head chunk."""
      dk_q, dv_q, dq_q = q_head_carry
      q_sub_head_start = q_head_idx * block_q_head
      q_head_start = kv_head_start * head_multiplier + q_sub_head_start
      q_head_indices = (
          (kv_head_start, q_sub_head_start)
          if multi_kv_head_tile
          else (q_head_start,)
      )

      def k_block_kernel(i, k_carry):
        """Processes one key/value block across all query blocks."""
        dk_k, dv_k, dq_k = k_carry
        kv_slice_start = i * block_kv
        start_indices_k = (0,) * len(batch_dims) + (
            kv_head_start,
            kv_slice_start,
            0,
        )
        start_indices_v = (0,) * len(batch_dims) + (
            kv_head_start,
            kv_slice_start,
            0,
        )
        k_outer, v_outer, k_scaled_outer = slice_kv_block(
            start_indices_k, start_indices_v, unfused_dma=num_kv_blocks > 4
        )
        kv_segment_ids_block = (
            lax.dynamic_slice_in_dim(
                segment_ids.kv, kv_slice_start, block_kv, axis=-1
            )
            if segment_ids is not None
            else None
        )
        diag_after_offdiag = peel_diagonal
        dk_head_init = jnp.zeros(
            batch_dims + q_head_tile_dims + (block_kv, k.shape[-1]), dtype=dtype
        )
        dv_head_init = jnp.zeros(
            batch_dims + q_head_tile_dims + (block_kv, v.shape[-1]), dtype=dtype
        )
        start_j = i + 1 if peel_diagonal else 0

        def q_block_kernel(step, inner_carry):
          """Processes an off diagonal query block."""
          dk_head_inner, dv_head_inner, dq_inner = inner_carry
          j = (
              num_q_blocks - 1 - (step - start_j)
              if diag_after_offdiag
              else step
          )
          q_slice_start = j * block_q
          start_indices_q = (
              (0,) * len(batch_dims) + q_head_indices + (q_slice_start, 0)
          )
          start_indices_log = (
              (0,) * len(batch_dims) + q_head_indices + (q_slice_start,)
          )
          q_block = jax.lax.dynamic_slice(
              q_work,
              tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_q),
              slice_sizes_q,
          )
          do_block = jax.lax.dynamic_slice(
              do_work,
              tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_q),
              slice_sizes_do,
          )
          lse_block = (
              jax.lax.dynamic_slice(
                  lse_work,
                  tuple(
                      jnp.asarray(i, dtype=jnp.int32) for i in start_indices_log
                  ),
                  slice_sizes_log,
              )
              .reshape((-1, block_q))
              .astype(jnp.float32)
              .reshape(slice_sizes_log + (1,))
          )
          di_block = (
              jax.lax.dynamic_slice(
                  di_work,
                  tuple(
                      jnp.asarray(i, dtype=jnp.int32) for i in start_indices_log
                  ),
                  slice_sizes_log,
              )
              .reshape((-1, block_q))
              .astype(jnp.float32)
              .reshape(slice_sizes_log + (1,))
          )
          dq_slice = jax.lax.dynamic_slice(
              dq_inner,
              tuple(jnp.asarray(i, dtype=jnp.int32) for i in start_indices_q),
              slice_sizes_q,
          )
          q_segment_ids_block = (
              lax.dynamic_slice_in_dim(
                  segment_ids.q, q_slice_start, block_q, axis=-1
              )
              if segment_ids is not None
              else None
          )
          mask_block_qkv = slice_mask_tile(
              kv_head_start,
              q_sub_head_start,
              q_head_start,
              q_slice_start,
              kv_slice_start,
          )
          return eval_bwd_tile_fused(
              q_block,
              k_outer,
              v_outer,
              do_block,
              lse_block,
              di_block,
              mask_block_qkv,
              q_segment_ids_block,
              kv_segment_ids_block,
              is_causal_diagonal=False,
              dk_prev=dk_head_inner,
              dv_prev=dv_head_inner,
              dq_prev=dq_slice,
              accumulate_dq=False,
              dq_carry_inner=dq_inner,
              q_slice_start=q_slice_start,
              kv_slice_start=kv_slice_start,
              q_head_indices=q_head_indices,
              fuse_id='12',
              k_scaled_tile=k_scaled_outer,
          )

        if num_q_blocks > 1 or not peel_diagonal:
          dk_head_block, dv_head_block, dq_k = _jax_k_loop(
              start_j,
              num_q_blocks,
              q_block_kernel,
              (dk_head_init, dv_head_init, dq_k),
          )
        else:
          dk_head_block, dv_head_block = (dk_head_init, dv_head_init)
        if diag_after_offdiag:
          start_indices_diag_q = (
              (0,) * len(batch_dims) + q_head_indices + (kv_slice_start, 0)
          )
          start_indices_diag_log = (
              (0,) * len(batch_dims) + q_head_indices + (kv_slice_start,)
          )
          q_diag = jax.lax.dynamic_slice(
              q_work,
              tuple(
                  jnp.asarray(i, dtype=jnp.int32) for i in start_indices_diag_q
              ),
              slice_sizes_q,
          )
          do_diag = jax.lax.dynamic_slice(
              do_work,
              tuple(
                  jnp.asarray(i, dtype=jnp.int32) for i in start_indices_diag_q
              ),
              slice_sizes_do,
          )
          lse_diag = (
              jax.lax.dynamic_slice(
                  lse_work,
                  tuple(
                      jnp.asarray(i, dtype=jnp.int32)
                      for i in start_indices_diag_log
                  ),
                  slice_sizes_log,
              )
              .reshape((-1, block_q))
              .astype(jnp.float32)
              .reshape(slice_sizes_log + (1,))
          )
          di_diag = (
              jax.lax.dynamic_slice(
                  di_work,
                  tuple(
                      jnp.asarray(i, dtype=jnp.int32)
                      for i in start_indices_diag_log
                  ),
                  slice_sizes_log,
              )
              .reshape((-1, block_q))
              .astype(jnp.float32)
              .reshape(slice_sizes_log + (1,))
          )
          dq_diag_slice = jax.lax.dynamic_slice(
              dq_k,
              tuple(
                  jnp.asarray(i, dtype=jnp.int32) for i in start_indices_diag_q
              ),
              slice_sizes_q,
          )
          mask_diag = slice_mask_tile(
              kv_head_start,
              q_sub_head_start,
              q_head_start,
              kv_slice_start,
              kv_slice_start,
          )
          q_diag_seg = (
              lax.dynamic_slice_in_dim(
                  segment_ids.q, kv_slice_start, block_q, axis=-1
              )
              if segment_ids is not None
              else None
          )
          dk_head_block, dv_head_block, dq_k = eval_bwd_tile_fused(
              q_diag,
              k_outer,
              v_outer,
              do_diag,
              lse_diag,
              di_diag,
              mask_diag,
              q_diag_seg,
              kv_segment_ids_block,
              is_causal_diagonal=True,
              dk_prev=dk_head_block,
              dv_prev=dv_head_block,
              dq_prev=dq_diag_slice,
              accumulate_dq=False,
              dq_carry_inner=dq_k,
              q_slice_start=kv_slice_start,
              kv_slice_start=kv_slice_start,
              q_head_indices=q_head_indices,
              fuse_id='11',
              k_scaled_tile=k_scaled_outer,
          )
        dk_block = (
            jnp.sum(dk_head_block.astype(jnp.float32), axis=-3)
            .astype(dtype)
            .reshape(slice_sizes_k)
        )
        dv_block = (
            jnp.sum(dv_head_block.astype(jnp.float32), axis=-3)
            .astype(dtype)
            .reshape(slice_sizes_v)
        )
        if num_q_head_blocks == 1:
          dk_k = lax.dynamic_update_slice(dk_k, dk_block, start_indices_k)
          dv_k = lax.dynamic_update_slice(dv_k, dv_block, start_indices_v)
        else:
          dk_k = lax.dynamic_update_slice(
              dk_k,
              dk_block
              + jax.lax.dynamic_slice(
                  dk_k,
                  tuple(
                      jnp.asarray(i, dtype=jnp.int32) for i in start_indices_k
                  ),
                  slice_sizes_k,
              ),
              start_indices_k,
          )
          dv_k = lax.dynamic_update_slice(
              dv_k,
              dv_block
              + jax.lax.dynamic_slice(
                  dv_k,
                  tuple(
                      jnp.asarray(i, dtype=jnp.int32) for i in start_indices_v
                  ),
                  slice_sizes_v,
              ),
              start_indices_v,
          )
        return (dk_k, dv_k, dq_k)

      if num_kv_blocks == 1:
        return k_block_kernel(0, (dk_q, dv_q, dq_q))
      return _jax_k_loop(0, num_kv_blocks, k_block_kernel, (dk_q, dv_q, dq_q))

    if num_q_head_blocks == 1:
      dk_kv, dv_kv, dq_kv = q_head_kernel(0, (dk_kv, dv_kv, dq_kv))
    else:
      dk_kv, dv_kv, dq_kv = _jax_k_loop(
          0, num_q_head_blocks, q_head_kernel, (dk_kv, dv_kv, dq_kv)
      )
    return (dk_kv, dv_kv, dq_kv)

  if num_q_head_blocks == 1:
    dk = lax.empty(k.shape, dtype=dtype)
    dv = lax.empty(v.shape, dtype=dtype)
  else:
    dk = jnp.zeros(k.shape, dtype=dtype)
    dv = jnp.zeros(v.shape, dtype=dtype)
  dq = jnp.zeros(q_work.shape, dtype=dtype)
  if num_kv_head_blocks == 1:
    dk, dv, dq = kv_head_kernel(0, (dk, dv, dq))
  else:
    dk, dv, dq = _jax_k_loop(
        0, num_kv_head_blocks, kv_head_kernel, (dk, dv, dq)
    )
  if multi_kv_head_tile:
    dq = dq.reshape(q.shape)
  return (dq, dk, dv)


_splash_attention_gqa_bwd_jax_k_jit = jax.jit(
    _splash_attention_gqa_bwd_jax_k_impl,
    static_argnames=[
        'block_q',
        'block_kv',
        'block_q_head',
        'block_kv_head',
        'dtype',
        'is_causal',
        'use_log2',
        'fuse_dus',
        'valid_q_len',
        'valid_kv_len',
        'q_prescaled_log2e',
    ],
)


def splash_attention_gqa_bwd_jax_k(
    q: Any,
    k: Any,
    v: Any,
    o: Any,
    do: Any,
    logsumexp: Any,
    segment_ids: Any,
    mask: Any,
    block_q: int,
    block_kv: int,
    block_q_head: int,
    block_kv_head: int,
    mask_value: Any,
    attn_logits_soft_cap: Any = None,
    dtype: Any = jnp.float32,
    is_causal: Any = False,
    use_log2: Any = False,
    di: Any = None,
    fuse_dus: Any = True,
    valid_q_len: Any = None,
    valid_kv_len: Any = None,
    q_offset: Any = None,
    q_prescaled_log2e: Any = False,
) -> Any:
  """Dispatches jax_k GQA backward pass, disabling fused DUS under `jax.vmap`."""
  array_args = (q, k, v, o, do, logsumexp, segment_ids, mask, di, q_offset)

  def _invoke_gqa(
      q_a,
      k_a,
      v_a,
      o_a,
      do_a,
      lse_a,
      seg_a,
      mask_a,
      di_a,
      qoff_a,
      dus_flag: Any,
  ):
    staged = _has_staged_tracer((
        q_a,
        k_a,
        v_a,
        o_a,
        do_a,
        lse_a,
        seg_a,
        mask_a,
        di_a,
        qoff_a,
        mask_value,
        attn_logits_soft_cap,
    ))
    fn = (
        _splash_attention_gqa_bwd_jax_k_impl
        if staged
        else _splash_attention_gqa_bwd_jax_k_jit
    )
    return fn(
        q_a,
        k_a,
        v_a,
        o_a,
        do_a,
        lse_a,
        seg_a,
        mask_a,
        block_q,
        block_kv,
        block_q_head,
        block_kv_head,
        mask_value,
        attn_logits_soft_cap,
        dtype,
        is_causal,
        use_log2,
        di_a,
        dus_flag,
        valid_q_len,
        valid_kv_len,
        qoff_a,
        q_prescaled_log2e,
    )

  if not fuse_dus or _has_batch_tracer(
      (array_args, mask_value, attn_logits_soft_cap)
  ):
    return _invoke_gqa(
        q, k, v, o, do, logsumexp, segment_ids, mask, di, q_offset, False
    )

  def _run_gqa(dus_flag: Any, args_tuple: Any):
    q_a, k_a, v_a, o_a, do_a, lse_a, seg_a, mask_a, di_a, qoff_a = args_tuple
    return _invoke_gqa(
        q_a, k_a, v_a, o_a, do_a, lse_a, seg_a, mask_a, di_a, qoff_a, dus_flag
    )

  @jax.custom_batching.custom_vmap
  def _call_gqa(args_tuple):
    return _run_gqa(True, args_tuple)

  @_call_gqa.def_vmap
  def _call_gqa_vmap(axis_size, in_batched, args_tuple):
    del axis_size
    if not any(jax.tree_util.tree_leaves(in_batched)):
      return (_run_gqa(False, args_tuple), (False, False, False))
    in_axes = (
        jax.tree_util.tree_map(lambda b: 0 if b else None, in_batched[0]),
    )
    out = jax.vmap(lambda a: _run_gqa(False, a), in_axes=in_axes)(args_tuple)
    return (out, (True, True, True))

  return _call_gqa(array_args)


def _pad_axis(x: Any, axis: int, pad_len: int, pad_val: Any) -> Any:
  """Pads `x` at the end of `axis` by `pad_len` elements with `pad_val`."""
  if pad_len == 0:
    return x
  norm_axis = axis if axis >= 0 else x.ndim + axis
  pads = [(0, 0)] * x.ndim
  pads[norm_axis] = (0, pad_len)
  return jnp.pad(x, pads, mode='constant', constant_values=pad_val)


def _select_effective_block_size(seq_len: int, block_size: int) -> int:
  """Selects an effective tile size for `seq_len` given requested `block_size`."""
  if seq_len >= block_size:
    return block_size
  if seq_len % 128 == 0:
    return seq_len
  return min(block_size, (seq_len + 127) // 128 * 128)


def splash_attention_bwd_jax_k(
    q: Any,
    k: Any,
    v: Any,
    o: Any,
    do: Any,
    logsumexp: Any,
    segment_ids: Any = None,
    mask: Any = None,
    *,
    block_q: int = 1024,
    block_kv: int = 1024,
    block_q_head: Any = None,
    block_kv_head: Any = None,
    is_mqa: Any = False,
    is_causal: Any = True,
    use_log2: Any = True,
    mask_value: Any = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: Any = None,
    dtype: Any = None,
    di: Any = None,
    fuse_dus: Any = True,
    q_offset: Any = None,
    q_prescaled_log2e: Any = False,
) -> Any:
  """Unified entry point for jax_k Splash Attention backward pass.

  Automatically inspects tensor shapes and head ratios to dispatch to either:
    1. `splash_attention_gqa_bwd_jax_k`: when `is_mqa=True` or `num_kv_heads <
    num_q_heads`.
       Handles rank 3 MQA tensors by expanding a singleton head dimension,
       executing GQA, and squeezing the output gradients back to rank 3.
    2. `splash_attention_mha_bwd_jax_k`: when `num_q_heads == num_kv_heads`.

  Non divisible sequence lengths are transparently padded to a multiple of the
  effective tile size, masked out along the padded positions inside the tile
  fusion, and sliced back to the original input shapes.

  Args:
    q: Query tensor of shape `(..., num_q_heads, q_seq_len, head_dim)`.
    k: Key tensor of shape `(..., num_kv_heads, kv_seq_len, head_dim)` or `(...,
      kv_seq_len, head_dim)` for MQA.
    v: Value tensor of shape `(..., num_kv_heads, kv_seq_len, head_dim)` or
      `(..., kv_seq_len, head_dim)` for MQA.
    o: Forward attention output of shape `(..., num_q_heads, q_seq_len,
      head_dim)`.
    do: Upstream gradient of shape `(..., num_q_heads, q_seq_len, head_dim)`.
    logsumexp: Forward logsumexp normalizer of shape `(..., num_q_heads,
      q_seq_len)`.
    segment_ids: Optional document segment boundary IDs for masking.
    mask: Optional attention mask array.
    block_q: Query sequence tile size.
    block_kv: Key/value sequence tile size.
    block_q_head: Query head block size for GQA or MHA.
    block_kv_head: Key/value head block size for GQA or MHA.
    is_mqa: Whether inputs follow Multi Query Attention (singleton KV head).
    is_causal: Whether lower triangular causal masking applies.
    use_log2: Whether base 2 exponentiation is used.
    mask_value: Additive value applied to masked positions before
      exponentiation.
    attn_logits_soft_cap: Optional threshold for tanh logit soft capping.
    dtype: Computation precision dtype. Defaults to `q.dtype`.
    di: Optional precomputed inner product sum(o * do, axis=-1).
    fuse_dus: Whether to fuse DynamicUpdateSlice inside the mega fusion.
    q_offset: Optional query position offset for causal masking.
    q_prescaled_log2e: Whether query tensor is already scaled by log2(e).

  Returns:
    A tuple of `(dq, dk, dv)` gradient arrays with the same shapes and
    precisions as the input `(q, k, v)` tensors.
  """
  if block_q <= 0 or block_kv <= 0:
    raise ValueError(
        f'block_q and block_kv must be positive, got block_q={block_q!r},'
        f' block_kv={block_kv!r}.'
    )
  if q_prescaled_log2e and (not use_log2):
    raise ValueError('q_prescaled_log2e=True requires use_log2=True.')
  eff_dtype = dtype if dtype is not None else q.dtype
  orig_q_seq_len = q.shape[-2]
  orig_kv_seq_len = k.shape[-2]
  num_q_heads = q.shape[-3]
  num_kv_heads = 1 if is_mqa else k.shape[-3]
  if num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f'num_q_heads ({num_q_heads}) must be divisible by num_kv_heads'
        f' ({num_kv_heads}).'
    )
  if mask is not None:
    _validate_mask_shape(mask, q.shape, orig_kv_seq_len)
  use_gqa_kernel = is_mqa or num_kv_heads < num_q_heads
  eff_bq = _select_effective_block_size(orig_q_seq_len, block_q)
  eff_bkv = _select_effective_block_size(orig_kv_seq_len, block_kv)
  q_rem = orig_q_seq_len % eff_bq
  kv_rem = orig_kv_seq_len % eff_bkv
  pad_q = eff_bq - q_rem if q_rem != 0 else 0
  pad_kv = eff_bkv - kv_rem if kv_rem != 0 else 0
  valid_q_len = orig_q_seq_len if pad_q > 0 else None
  valid_kv_len = orig_kv_seq_len if pad_kv > 0 else None
  q_in = _pad_axis(q, -2, pad_q, 0)
  k_in = _pad_axis(k, -2, pad_kv, 0)
  v_in = _pad_axis(v, -2, pad_kv, 0)
  o_in = _pad_axis(o, -2, pad_q, 0)
  do_in = _pad_axis(do, -2, pad_q, 0)
  lse_in = _pad_axis(logsumexp, -1, pad_q, 0)
  di_in = _pad_axis(di, -1, pad_q, 0) if di is not None else None
  seg_in = (
      SegmentIds(
          q=_pad_axis(segment_ids.q, -1, pad_q, 0),
          kv=_pad_axis(segment_ids.kv, -1, pad_kv, 0),
      )
      if segment_ids is not None
      else None
  )
  mask_in = (
      _pad_axis(_pad_axis(mask, -2, pad_q, False), -1, pad_kv, False)
      if mask is not None
      else None
  )
  if use_gqa_kernel:
    squeeze_kv_head = is_mqa and k_in.ndim == q_in.ndim - 1
    k_gqa = jnp.expand_dims(k_in, axis=-3) if squeeze_kv_head else k_in
    v_gqa = jnp.expand_dims(v_in, axis=-3) if squeeze_kv_head else v_in
    head_multiplier = num_q_heads // num_kv_heads
    eff_bq_head = (
        min(block_q_head, head_multiplier)
        if block_q_head is not None
        else head_multiplier
    )
    eff_bkv_head = (
        min(block_kv_head, num_kv_heads) if block_kv_head is not None else 1
    )
    if eff_bq_head <= 0 or head_multiplier % eff_bq_head != 0:
      raise ValueError(
          f'block_q_head ({eff_bq_head}) must divide head_multiplier'
          f' ({head_multiplier}).'
      )
    if eff_bkv_head <= 0 or num_kv_heads % eff_bkv_head != 0:
      raise ValueError(
          f'block_kv_head ({eff_bkv_head}) must divide num_kv_heads'
          f' ({num_kv_heads}).'
      )
    dq_out, dk_out, dv_out = splash_attention_gqa_bwd_jax_k(
        q=q_in,
        k=k_gqa,
        v=v_gqa,
        o=o_in,
        do=do_in,
        logsumexp=lse_in,
        segment_ids=seg_in,
        mask=mask_in,
        block_q=eff_bq,
        block_kv=eff_bkv,
        block_q_head=eff_bq_head,
        block_kv_head=eff_bkv_head,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        dtype=eff_dtype,
        is_causal=is_causal,
        use_log2=use_log2,
        di=di_in,
        fuse_dus=fuse_dus,
        valid_q_len=valid_q_len,
        valid_kv_len=valid_kv_len,
        q_offset=q_offset,
        q_prescaled_log2e=q_prescaled_log2e,
    )
    if squeeze_kv_head:
      dk_out = jnp.squeeze(dk_out, axis=-3)
      dv_out = jnp.squeeze(dv_out, axis=-3)
  else:
    eff_b_head = (
        min(block_q_head, num_q_heads)
        if block_q_head is not None
        else num_q_heads
    )
    if eff_b_head <= 0 or num_q_heads % eff_b_head != 0:
      raise ValueError(
          f'block_q_head ({eff_b_head}) must divide num_q_heads'
          f' ({num_q_heads}).'
      )
    if (
        block_kv_head is not None
        and min(block_kv_head, num_kv_heads) != eff_b_head
    ):
      raise ValueError(
          f'For MHA, block_kv_head ({block_kv_head}) must match block_q_head'
          f' ({eff_b_head}).'
      )
    dq_out, dk_out, dv_out = splash_attention_mha_bwd_jax_k(
        q_in,
        k_in,
        v_in,
        o_in,
        do_in,
        lse_in,
        seg_in,
        mask_in,
        mask_value,
        attn_logits_soft_cap,
        eff_bq,
        eff_bkv,
        eff_b_head,
        eff_dtype,
        is_causal,
        use_log2,
        di_in,
        fuse_dus,
        valid_q_len,
        valid_kv_len,
        q_offset,
        q_prescaled_log2e,
    )
  if pad_q > 0:
    dq_out = dq_out[..., :orig_q_seq_len, :]
  if pad_kv > 0:
    dk_out = dk_out[..., :orig_kv_seq_len, :]
    dv_out = dv_out[..., :orig_kv_seq_len, :]
  if q_prescaled_log2e:
    dq_out = (dq_out.astype(jnp.float32) * LOG2E_INV).astype(q.dtype)
    dk_out = (dk_out.astype(jnp.float32) * LOG2E_INV).astype(k.dtype)
  return (dq_out, dk_out, dv_out)
