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

"""Stage 1 jax_k Fused Mixture of Experts (MoE) champion kernel."""

import functools
import inspect
import math
from typing import Any

import jax
from jax.experimental import layout as jax_layout
from jax.experimental import scheduling_groups as jax_sched
from jax.experimental import xla_metadata
import jax.numpy as jnp

from xla.jax_kernels.fused_moe import configs

# pylint: disable=invalid-name,missing-class-docstring,missing-function-docstring,broad-exception-caught,g-complex-comprehension,g-long-ternary
_JAX_K_LOOP_DEPTH = 0


def _jax_k_color_call(x: Any, *, memory_space: str, is_output: bool) -> Any:
  dev = jax.config.jax_default_device
  plat = dev.platform if dev is not None else jax.default_backend()
  if plat == "cpu":
    return x
  t = f"xla_color_{'prev_output' if is_output else 'next_input'}_{memory_space}"

  @jax.custom_jvp
  def _fn(v: Any) -> Any:
    return jax.ffi.ffi_call(
        t,
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        vmap_method="broadcast_all",
        custom_call_api_version=2,
        legacy_backend_config="",
    )(v)

  @_fn.defjvp
  def _fn_jvp(primals: Any, tangents: Any) -> Any:
    (v,), (dv,) = primals, tangents
    return _fn(v), _fn(dv)

  return _fn(x)


_JAX_K_HBM_SPILLS: set[int] = set()


def _jax_k_cp_ltag(src: Any, dst: Any) -> Any:
  lm = globals().get("_JAX_K_LMAP")
  if isinstance(lm, dict):
    en = lm.get(id(src))
    if en is not None and en[0] is src:
      lm[id(dst)] = (dst, en[1])
  return dst


def _jax_k_color(
    x, memory_space: str, *, is_output: bool = False, **clear_extra: Any
):
  if memory_space in ("hbm", "pinned_hbm") and is_output:
    jax.tree.map(
        lambda l: _JAX_K_HBM_SPILLS.add(id(l)) if l is not None else None, x
    )
    return x
  if _JAX_K_LOOP_DEPTH > 0 or globals().get("_JAX_K_PAD_DISC", False):
    return x
  with xla_metadata.set_xla_metadata(
      MUST_FUSE=None,
      MAXIMAL_FUSE=None,
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
      **clear_extra,
  ):
    return jax.tree.map(
        lambda l: _jax_k_cp_ltag(
            l,
            _jax_k_color_call(
                l, memory_space=memory_space, is_output=is_output
            ),
        ),
        x,
    )


def _jax_k_bind_spill(x: Any, **clear_extra: Any) -> Any:
  def _b(l: Any) -> Any:
    if l is None or id(l) not in _JAX_K_HBM_SPILLS:
      return l
    _JAX_K_HBM_SPILLS.discard(id(l))
    if _JAX_K_LOOP_DEPTH > 0 or globals().get("_JAX_K_PAD_DISC", False):
      return l
    with xla_metadata.set_xla_metadata(
        MUST_FUSE=None,
        MAXIMAL_FUSE=None,
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
        **clear_extra,
    ):
      return _jax_k_cp_ltag(
          l,
          _jax_k_color_call(l, memory_space="hbm", is_output=True),
      )

  return jax.tree.map(_b, x)


def _jax_k_fusion_spills(fn: Any) -> Any:
  return functools.wraps(fn)(
      lambda *a, **kw: fn(
          *jax.tree.map(_jax_k_bind_spill, a),
          **jax.tree.map(_jax_k_bind_spill, kw),
      )
  )


class _jax_k_layout(str):
  major_to_minor: tuple[int, ...] | None
  tiling: tuple[tuple[int, ...], ...] | None
  sub_byte_element_size_in_bits: int
  padless: bool
  rank_adaptive_swap_last_two: bool

  def __new__(
      cls,
      spec: Any = None,
      *,
      major_to_minor: Any = None,
      minor_to_major: Any = None,
      tiling: Any = None,
      tile_shape: Any = None,
      sublane_packing: Any = None,
      sub_byte_element_size_in_bits: int = 0,
      padless: bool = False,
      rank_adaptive_swap_last_two: bool = False,
  ):
    m2m, tl = None, None
    if isinstance(spec, _jax_k_layout):
      m2m, tl = spec.major_to_minor, spec.tiling
      padless = spec.padless or padless
      rank_adaptive_swap_last_two = (
          spec.rank_adaptive_swap_last_two or rank_adaptive_swap_last_two
      )
      sub_byte_element_size_in_bits = (
          sub_byte_element_size_in_bits or spec.sub_byte_element_size_in_bits
      )
    elif isinstance(spec, str):
      o, _, a = spec.strip().strip("{}").partition(":")
      m2m = (
          tuple(reversed([int(d) for d in o.split(",") if d.strip()]))
          if o.strip()
          else None
      )
      tl = (
          tuple(
              tuple(int(d) for d in p.rstrip(")").split(",") if d.strip())
              for p in a.split("T(")[1].split("(")
              if p.strip()
          )
          if "T(" in a
          else None
      )
    elif isinstance(spec, (tuple, list)) and all(
        isinstance(x, int) for x in spec
    ):
      m2m = tuple(int(x) for x in spec)
    elif spec is not None and hasattr(spec, "major_to_minor"):
      rm, rt = getattr(spec, "major_to_minor", None), getattr(
          spec, "tiling", None
      )
      m2m = tuple(int(x) for x in rm) if rm is not None else None
      tl = (
          tuple(tuple(int(d) for d in t) for t in rt)
          if rt is not None
          else None
      )
    if major_to_minor is not None:
      m2m = tuple(int(x) for x in major_to_minor)
    elif minor_to_major is not None:
      m2m = tuple(reversed(tuple(int(x) for x in minor_to_major)))
    if tiling is not None:
      tl = tuple(tuple(int(d) for d in t) for t in tiling)
    elif tile_shape is not None:
      tl = (tuple(int(d) for d in tile_shape),) + (
          (tuple(int(d) for d in sublane_packing),)
          if sublane_packing is not None
          else ()
      )
    o_str = ",".join(str(d) for d in reversed(m2m)) if m2m is not None else ""
    s_val = (
        o_str
        if padless or not tl
        else f"{o_str}:T"
        + "".join(f"({','.join(str(x) for x in t)})" for t in tl)
    )
    obj = super().__new__(cls, s_val)
    obj.major_to_minor, obj.tiling = m2m, tl
    obj.sub_byte_element_size_in_bits = int(sub_byte_element_size_in_bits)
    obj.padless, obj.rank_adaptive_swap_last_two = bool(padless), bool(
        rank_adaptive_swap_last_two
    )
    return obj

  @property
  def minor_to_major(self) -> tuple[int, ...] | None:
    return (
        None
        if self.major_to_minor is None
        else tuple(reversed(self.major_to_minor))
    )

  def resolve_major_to_minor(
      self, ndim: int | None = None
  ) -> tuple[int, ...] | None:
    if self.major_to_minor is not None:
      if ndim is None or len(self.major_to_minor) == ndim:
        return self.major_to_minor
      if self.rank_adaptive_swap_last_two and ndim >= 2:
        return tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
      if ndim > len(self.major_to_minor):
        return tuple(range(ndim - len(self.major_to_minor))) + tuple(
            d + ndim - len(self.major_to_minor) for d in self.major_to_minor
        )
      return self.major_to_minor
    if self.rank_adaptive_swap_last_two and ndim is not None and ndim >= 2:
      return tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
    return tuple(range(ndim)) if ndim is not None else None

  def to_jax_layout(self, ndim: int | None = None) -> Any:
    kw: dict[str, Any] = {}
    m2m = self.resolve_major_to_minor(ndim=ndim)
    if m2m is not None:
      kw["major_to_minor"] = m2m
    if (
        not self.padless
        and self.tiling is not None
        and jax.default_backend() == "tpu"
    ):
      kw["tiling"] = self.tiling
    if self.sub_byte_element_size_in_bits and jax.default_backend() == "tpu":
      kw["sub_byte_element_size_in_bits"] = self.sub_byte_element_size_in_bits
    return jax_layout.Layout(**kw)

  def to_format(
      self, sharding_or_device: Any = None, ndim: int | None = None
  ) -> Any:
    sh = (
        jax.sharding.SingleDeviceSharding(jax.devices()[0])
        if sharding_or_device is None
        else (
            sharding_or_device
            if isinstance(sharding_or_device, jax.sharding.Sharding)
            else jax.sharding.SingleDeviceSharding(sharding_or_device)
        )
    )
    return jax_layout.Format(self.to_jax_layout(ndim=ndim), sh)


_jax_k_layout.BATCH_IN_LANES_BF16 = _jax_k_layout(
    major_to_minor=(0, 1, 2, 4, 3),
    tiling=((16, 128), (2, 1)),
    rank_adaptive_swap_last_two=True,
)
_jax_k_layout.BATCH_IN_LANES_AUG_BF16 = _jax_k_layout(
    major_to_minor=(0, 1, 2, 4, 3),
    tiling=((8, 128), (2, 1)),
    rank_adaptive_swap_last_two=True,
)
_jax_k_layout.SEQ_MINOR_4D_BF16 = _jax_k_layout(
    major_to_minor=(0, 1, 3, 2),
    tiling=((16, 128), (2, 1)),
    rank_adaptive_swap_last_two=True,
)
_jax_k_layout.SEQ_MINOR_4D_AUG_BF16 = _jax_k_layout(
    major_to_minor=(0, 1, 3, 2),
    tiling=((8, 128), (2, 1)),
    rank_adaptive_swap_last_two=True,
)
_jax_k_layout.MXU_TILED = _jax_k_layout(tiling=((8, 128),))
_JAX_K_LMAP: dict[int, tuple[Any, Any]] = {}


def _jax_k_set_ltag(v: Any, lt: Any) -> Any:
  if v is not None and lt is not None:
    _JAX_K_LMAP[id(v)] = (v, lt)
  return v


def _jax_k_get_ltag(v: Any) -> Any:
  if v is None:
    return None
  en = _JAX_K_LMAP.get(id(v))
  return en[1] if en is not None and en[0] is v else None


def _jax_k_with_layout(
    x: Any,
    major_to_minor: Any = None,
    tiling: Any = None,
    *,
    minor_to_major: Any = None,
    padless: bool = False,
    **clear_extra: Any,
) -> Any:
  nl = (
      major_to_minor
      if isinstance(major_to_minor, _jax_k_layout)
      and tiling is None
      and minor_to_major is None
      and not padless
      else (
          None
          if hasattr(major_to_minor, "major_to_minor")
          and not isinstance(major_to_minor, _jax_k_layout)
          else _jax_k_layout(
              major_to_minor,
              minor_to_major=minor_to_major,
              tiling=tiling,
              padless=padless,
          )
      )
  )
  lt = nl if nl is not None else major_to_minor

  def _cl(l: Any) -> Any:
    if l is None or not hasattr(l, "shape"):
      return l
    if globals().get("_JAX_K_PAD_DISC", False):
      return _jax_k_set_ltag(l, lt)
    return _jax_k_set_ltag(
        jax_layout.with_layout_constraint(
            l, major_to_minor if nl is None else nl.to_jax_layout(ndim=l.ndim)
        ),
        lt,
    )

  with xla_metadata.set_xla_metadata(
      MUST_FUSE=None,
      MAXIMAL_FUSE=None,
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
      **clear_extra,
  ):
    return jax.tree.map(_cl, x)


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
          ",".join(str(x) for x in v)
          if isinstance(v, (tuple, list))
          else str(v)
      )
  )


def _jax_k_fusion_meta(
    *, must_fuse: bool = False, fusion_group: Any = None, **kw: Any
) -> dict[str, str]:
  out: dict[str, str] = {}
  if must_fuse:
    out["MUST_FUSE"] = (
        _jax_k_fmt_meta(fusion_group)
        if fusion_group is not None
        else _jax_k_next_fuid(1000)
    )
  for k, m in (
      ("scoped_vmem_limit_kib", "scoped_vmem_limit_kib"),
      ("buffering_level", "buffering_level"),
      ("output_window_bounds", "output_window_bounds"),
      ("kernel_window_bounds", "kernel_window_bounds"),
      ("input_window_bounds", "input_window_bounds"),
      ("integer", "integer"),
      ("fuse_limit", "FUSE_LIMIT"),
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
      "scoped_vmem_limit_kib" in extra
      and extra["scoped_vmem_limit_kib"] is not None
  )
  sv_imp = (
      None
      if sv_exp
      else (vmem_kib if isinstance(vmem_kib, int) and vmem_kib >= 512 else None)
  )
  base_cm: dict[str, str] = {}
  for k, v in (
      ("scoped_vmem_limit_kib", scoped_vmem_limit_kib),
      ("buffering_level", buffering_level),
      ("output_window_bounds", owb),
      ("kernel_window_bounds", kernel_window_bounds),
      ("input_window_bounds", input_window_bounds),
      ("integer", integer),
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
    base_cm["MUST_FUSE"] = _jax_k_fmt_meta(fusion_group)
  if isinstance(unsafe_metadata, dict):
    for k, v in unsafe_metadata.items():
      if v is not None:
        base_cm[str(k)] = _jax_k_fmt_meta(v)
  base_im: dict[str, Any] = dict({
      "MUST_FUSE": None,
      "MAXIMAL_FUSE": None,
      "FUSE_LIMIT": None,
      "scoped_vmem_limit_kib": None,
      "buffering_level": None,
      "output_window_bounds": None,
      "kernel_window_bounds": None,
      "input_window_bounds": None,
      "integer": None,
      "xla_disable_while_loop_copies": None,
      "xla_while_loop_async_pipelining_levels": None,
      "xla_pipeline_loop_carried_dus": None,
      "xla_while_loop_async_immediate_revisiting": None,
      "xla_post_fusion_pipelined_loop": None,
      "xla_disjoint_read_write_regions": None,
      "slice_sizes": None,
      "ColorToVMEM": None,
      "ColorInputToHBM": None,
      "explicit_layout": None,
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
      bs = globals().get("_jax_k_bind_spill")
      if bs is not None:
        a, kw = jax.tree.map(bs, a), jax.tree.map(bs, kw)
      cm = dict(base_cm)
      if sv_imp is not None and not _has_ret_layout:
        v_ins = [
            x
            for idx, arg in enumerate(a)
            if idx not in _non_vmem_idxs
            for x in jax.tree.leaves(arg)
            if hasattr(x, "shape")
        ] + [
            x
            for kn, kv in kw.items()
            if kn not in _non_vmem_names
            for x in jax.tree.leaves(kv)
            if hasattr(x, "shape")
        ]
        if len(v_ins) < 2 and all(
            getattr(getattr(x, "parent", None), "primitive", None) is not None
            and getattr(getattr(x, "parent", None), "primitive").name
            == "dynamic_slice"
            for x in v_ins
        ):
          try:
            if "LayoutConstraint" not in str(jax.make_jaxpr(f)(*a, **kw)):
              cm["scoped_vmem_limit_kib"] = _jax_k_fmt_meta(sv_imp)
          except Exception:
            pass
      im = dict(base_im)
      if must_fuse:
        if fusion_group is None:
          im["MUST_FUSE"] = uid
      else:
        im["MAXIMAL_FUSE"] = (
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


P = jax.sharding.PartitionSpec
cdiv = configs.cdiv
align_to = configs.align_to
get_dtype_packing = configs.get_dtype_packing
get_tuned_block_sizes = configs.get_tuned_block_sizes
FusedMoeTilingConfig = configs.FusedMoeTilingConfig


def _preserve_dtype_rounding(x: Any, dtype: Any) -> Any:
  """Preserves exact bfloat16 rounding across XLA fusion passes."""
  if jnp.dtype(dtype) == jnp.bfloat16:
    return jax.lax.reduce_precision(
        x.astype(jnp.float32), exponent_bits=8, mantissa_bits=7
    ).astype(jnp.bfloat16)
  return x.astype(dtype)


def apply_scoring_fn(
    scoring_fn: Any, x: Any, *, use_mosaic_exp: Any = False
) -> Any:
  """Applies router scoring function along the expert dimension."""
  match scoring_fn:
    case "softmax":
      if use_mosaic_exp and x.dtype == jnp.bfloat16:
        x_f32 = jax.lax.reduce_precision(
            x.astype(jnp.float32), exponent_bits=8, mantissa_bits=7
        )
        x_max = jnp.max(x_f32, axis=-1, keepdims=True)
        shifted = jax.lax.reduce_precision(
            x_f32 - x_max, exponent_bits=8, mantissa_bits=7
        )
        u = jax.lax.reduce_precision(
            shifted * jnp.float32(1.4453125), exponent_bits=8, mantissa_bits=7
        )
        unnorm = jax.lax.reduce_precision(
            jnp.exp2(u), exponent_bits=8, mantissa_bits=7
        )
        denom = jax.lax.reduce_precision(
            jnp.sum(unnorm, axis=-1, keepdims=True),
            exponent_bits=8,
            mantissa_bits=7,
        )
        rhs_inv = jax.lax.reduce_precision(
            jnp.float32(1.0) / denom, exponent_bits=8, mantissa_bits=7
        )
        return jax.lax.reduce_precision(
            unnorm * rhs_inv, exponent_bits=8, mantissa_bits=7
        ).astype(jnp.bfloat16)
      return _preserve_dtype_rounding(jax.nn.softmax(x, axis=-1), x.dtype)
    case "sigmoid":
      if x.dtype == jnp.bfloat16:
        x_f32 = jax.lax.reduce_precision(
            x.astype(jnp.float32), exponent_bits=8, mantissa_bits=7
        )
        u = jax.lax.reduce_precision(
            -x_f32 * jnp.float32(1.4453125), exponent_bits=8, mantissa_bits=7
        )
        exp_neg = jax.lax.reduce_precision(
            jnp.exp2(u), exponent_bits=8, mantissa_bits=7
        )
        denom = jax.lax.reduce_precision(
            jnp.float32(1.0) + exp_neg, exponent_bits=8, mantissa_bits=7
        )
        return jax.lax.reduce_precision(
            jnp.float32(1.0) / denom, exponent_bits=8, mantissa_bits=7
        ).astype(jnp.bfloat16)
      return (1.0 / (1.0 + jnp.exp(-x))).astype(x.dtype)
    case _:
      raise NotImplementedError(f"Unsupported scoring function: {scoring_fn}")


jax.config.update("jax_use_hlo_logistic_lowering", True)


def _hlo_logistic(x: Any) -> Any:
  """Lowers logistic directly to stablehlo.logistic (TPU vsigshft)."""
  return jax.lax.logistic(x)


def swigluoai(
    gate: Any, up: Any, *, alpha: Any = 1.702, limit: Any = 7.0
) -> Any:
  """Clamped SwiGLU activation used in models such as GPT OSS."""
  limit_val = jnp.asarray(limit, dtype=gate.dtype)
  gate = jax.lax.min(gate, limit_val)
  up = jax.lax.clamp(-limit_val, up, limit_val)
  glu = gate * _hlo_logistic(alpha * gate)
  return (up + 1.0) * glu


def _apply_gate_fn(acc1: Any, act_fn: Any) -> Any:
  """Applies the unary gate activation to the gate projection acc1."""
  if act_fn == "silu":
    return acc1 * _hlo_logistic(acc1)
  elif act_fn == "gelu":
    return jax.nn.gelu(acc1)
  elif act_fn == "swigluoai":
    gate = jax.lax.min(acc1, jnp.asarray(7.0, dtype=acc1.dtype))
    return gate * _hlo_logistic(1.702 * gate)
  else:
    raise NotImplementedError(f"Unsupported activation function: {act_fn}")


def _combine_gate_up(gate: Any, acc3: Any, act_fn: Any) -> Any:
  """Combines activated gate projection with up projection acc3."""
  if act_fn in ("silu", "gelu"):
    return gate * acc3
  elif act_fn == "swigluoai":
    limit_val = jnp.asarray(7.0, dtype=acc3.dtype)
    up = jax.lax.clamp(-limit_val, acc3, limit_val)
    return (up + 1.0) * gate
  else:
    raise NotImplementedError(f"Unsupported activation function: {act_fn}")


def apply_act_fn(acc1: Any, acc3: Any, act_fn: Any) -> Any:
  """Applies gated activation function to gate (acc1) and up (acc3) projections."""
  return _combine_gate_up(_apply_gate_fn(acc1, act_fn), acc3, act_fn)


def ref_moe(
    tokens: Any,
    w1: Any,
    w2: Any,
    gating_output: Any,
    top_k: int,
    *,
    renormalize_topk_logits: Any = False,
    act_fn: Any = "silu",
    scoring_fn: Any = "softmax",
    subc_quant_w1_sz: Any = None,
    subc_quant_w2_sz: Any = None,
    w1_scale: Any = None,
    w2_scale: Any = None,
    b1: Any = None,
    b2: Any = None,
) -> Any:
  """Reference JAX implementation of Fused MoE."""
  n_tokens = tokens.shape[0]
  gating_logits = apply_scoring_fn(scoring_fn, gating_output)
  top_k_logits, top_k_indices = jax.lax.top_k(gating_logits, top_k)
  if renormalize_topk_logits:
    top_k_logits = top_k_logits / jnp.sum(top_k_logits, axis=-1, keepdims=True)
  t_outputs = []
  hidden_size, intermediate_size = w1.shape[-2:]
  for i in range(n_tokens):
    curr_token = jnp.expand_dims(tokens[i], axis=0)
    assigned_expert_ids = top_k_indices[i]
    tok_expert_act = []
    for expert_id in assigned_expert_ids:
      expert_w1 = w1[expert_id, 0].astype(jnp.float32)
      expert_w3 = w1[expert_id, 1].astype(jnp.float32)
      if w1_scale is not None:
        assert subc_quant_w1_sz is not None
        expert_w1 *= jnp.repeat(
            w1_scale[expert_id, 0, :, 0], subc_quant_w1_sz, axis=0
        )[:hidden_size]
        expert_w3 *= jnp.repeat(
            w1_scale[expert_id, 1, :, 0], subc_quant_w1_sz, axis=0
        )[:hidden_size]
      expert_weight_1 = jnp.concat([expert_w1, expert_w3], axis=-1)
      expert_weight_2 = w2[expert_id].astype(jnp.float32)
      if w2_scale is not None:
        assert subc_quant_w2_sz is not None
        expert_weight_2 *= jnp.repeat(
            w2_scale[expert_id, :, 0], subc_quant_w2_sz, axis=0
        )[:intermediate_size]
      gmm_1_out = curr_token @ expert_weight_1
      gmm1_w1_proj, gmm1_w3_proj = jnp.split(gmm_1_out, 2, axis=-1)
      if b1 is not None:
        gmm1_w1_proj += b1[expert_id : expert_id + 1, 0, 0]
        gmm1_w3_proj += b1[expert_id : expert_id + 1, 1, 0]
      act = apply_act_fn(gmm1_w1_proj, gmm1_w3_proj, act_fn)
      gmm_2_out = act @ expert_weight_2
      if b2 is not None:
        gmm_2_out += b2[expert_id : expert_id + 1, 0]
      tok_expert_act.append(gmm_2_out)
    experts_act = jnp.concatenate(tok_expert_act, axis=0)
    top_k_weights = jnp.expand_dims(top_k_logits[i], axis=1)
    weighted_output = jnp.sum(
        experts_act * top_k_weights, axis=0, keepdims=True
    )
    t_outputs.append(weighted_output.astype(tokens.dtype))
  return jnp.concatenate(t_outputs, axis=0)


def _lock_compact_layout(x: Any) -> Any:
  """Constrains a 16 bit or 32 bit tensor to compact (8, 128) tiling."""
  if x is None or not hasattr(x, "shape") or x.ndim < 2:
    return x
  bits = configs.itemsize_bits(x.dtype)
  if bits < 16 or bits > 32:
    return x
  packing = max(1, 32 // bits)
  tiling: tuple[tuple[int, ...], ...] = (
      ((8, 128),) if packing == 1 else ((8, 128), (packing, 1))
  )
  space = None
  out = _jax_k_with_layout(
      x, major_to_minor=tuple(range(x.ndim)), tiling=tiling
  )
  return out if space is None else out


def _pin_to_hbm(x: Any) -> Any:
  """Pins an intermediate tensor to HBM with native tiling."""
  if x is None:
    return None
  x = _lock_compact_layout(x)
  return _jax_k_bind_spill(_jax_k_color(x, "hbm", is_output=True))


def _read_smem_scalar(arr: Any, idx: int) -> Any:
  """Extracts a scalar int32 from a 1D SMEM array at static index idx."""
  sliced = jax.lax.slice_in_dim(arr, idx, idx + 1, axis=0)
  return jnp.reshape(_jax_k_color(sliced, "smem", is_output=False), ())


def _select_word_tree(words: Any, word_idx: Any, lo: int, hi: int) -> Any:
  """Selects words[word_idx] using a binary scalar select tree in SREG."""
  if hi - lo <= 1:
    return words[lo]
  mid = (lo + hi) // 2
  left = _select_word_tree(words, word_idx, lo, mid)
  right = _select_word_tree(words, word_idx, mid, hi)
  return jnp.where(word_idx < jnp.int32(mid), left, right)


def _mosaic_argmax_axis1(work: Any, iota: Any) -> Any:
  """Computes axis 1 argmax matching TPU vmax.index.xlane tie breaking."""
  num_experts = work.shape[1]
  if num_experts <= 128:
    max_val = jnp.max(work, axis=1, keepdims=True)
    return jnp.max(
        jnp.where(work == max_val, iota, jnp.int32(-1)), axis=1, keepdims=True
    )
  num_tiles = (num_experts + 128 - 1) // 128
  pad_e = num_tiles * 128 - num_experts
  work_pad = (
      jnp.pad(work, ((0, 0), (0, pad_e)), constant_values=-jnp.inf)
      if pad_e > 0
      else work
  )
  if num_tiles == 2:
    t0 = work_pad[:, :128]
    t1 = work_pad[:, 128:256]
    lane_max = jnp.maximum(t0, t1)
    lane_iota = jax.lax.broadcasted_iota(jnp.int32, lane_max.shape, 1)
    lane_global_idx = jnp.where(t0 >= t1, lane_iota, lane_iota + jnp.int32(128))
    row_max = jnp.max(lane_max, axis=1, keepdims=True)
    win_lane = jnp.max(
        jnp.where(lane_max == row_max, lane_iota, jnp.int32(-1)),
        axis=1,
        keepdims=True,
    )
    return jnp.max(
        jnp.where(lane_iota == win_lane, lane_global_idx, jnp.int32(-1)),
        axis=1,
        keepdims=True,
    )
  work_tiles = work_pad.reshape(work.shape[0], num_tiles, 128)
  lane_max = jnp.max(work_tiles, axis=1)
  lane_tile_idx = jnp.argmax(work_tiles, axis=1).astype(jnp.int32)
  lane_iota = jax.lax.broadcasted_iota(jnp.int32, lane_max.shape, 1)
  lane_global_idx = lane_tile_idx * jnp.int32(128) + lane_iota
  row_max = jnp.max(lane_max, axis=1, keepdims=True)
  win_lane = jnp.max(
      jnp.where(lane_max == row_max, lane_iota, jnp.int32(-1)),
      axis=1,
      keepdims=True,
  )
  return jnp.take_along_axis(lane_global_idx, win_lane, axis=1)


def _compute_topk_routing(
    gating_padded: Any,
    num_experts: int,
    top_k: int,
    scoring_fn: Any,
    renormalize_topk_logits: Any,
    *,
    ep_axis_name: Any = None,
    ep_size: int = 1,
) -> Any:
  """Computes top k expert weights and indices matching Pallas tie breaking."""
  scores = apply_scoring_fn(scoring_fn, gating_padded, use_mosaic_exp=True)
  work = scores.astype(jnp.float32)[:, :num_experts]
  iota = jax.lax.broadcasted_iota(jnp.int32, work.shape, 1)
  if scores.dtype == jnp.bfloat16 and num_experts <= 256:
    payload = (
        (iota & jnp.int32(127)) << 9 | jnp.int32(1) - (iota >> 7) << 8 | iota
    )
    packed = (
        jax.lax.bitcast_convert_type(work, jnp.int32) & jnp.int32(2147418112)
        | payload
    )
    wins = []
    for k_id in range(top_k):
      win = jnp.max(packed, axis=1, keepdims=True)
      wins.append(win)
      if k_id != top_k - 1:
        packed = jnp.where(packed == win, jnp.int32(-1), packed)
    win_mat = jnp.concatenate(wins, axis=1)
    if ep_size > 1 and ep_axis_name is not None:
      win_mat = jax.lax.all_gather(
          win_mat, axis_name=ep_axis_name, tiled=True, axis=0
      )
    top_k_indices = win_mat & jnp.int32(255)
    top_k_weights = jax.lax.bitcast_convert_type(
        win_mat & jnp.int32(2147418112), jnp.float32
    )
    if renormalize_topk_logits:
      top_k_weights = top_k_weights / jnp.sum(
          top_k_weights, axis=1, keepdims=True
      )
    return (top_k_weights, top_k_indices)
  logits_list = []
  indices_list = []
  logits_sum = jnp.zeros((work.shape[0], 1), dtype=jnp.float32)
  for k_id in range(top_k):
    max_val = jnp.max(work, axis=1, keepdims=True)
    max_idx = _mosaic_argmax_axis1(work, iota)
    logits_list.append(max_val)
    indices_list.append(max_idx)
    if renormalize_topk_logits:
      logits_sum = logits_sum + max_val
    if k_id != top_k - 1:
      mask = iota == max_idx
      work = jnp.where(mask, -jnp.inf, work)
  if renormalize_topk_logits:
    logits_list = [v / logits_sum for v in logits_list]
  top_k_weights = jnp.concatenate(logits_list, axis=1)
  top_k_indices = jnp.concatenate(indices_list, axis=1)
  if ep_size > 1 and ep_axis_name is not None:
    packed_routing_local = jnp.concatenate(
        [jax.lax.bitcast_convert_type(top_k_weights, jnp.int32), top_k_indices],
        axis=1,
    )
    all_packed_routing = jax.lax.all_gather(
        packed_routing_local, axis_name=ep_axis_name, tiled=True, axis=0
    )
    top_k_weights = jax.lax.bitcast_convert_type(
        all_packed_routing[:, :top_k], jnp.float32
    )
    top_k_indices = all_packed_routing[:, top_k:]
  return (top_k_weights, top_k_indices)


def _prepare_expert_tiles(
    tokens: Any,
    top_k_indices: Any,
    top_k_weights: Any,
    local_num_experts: int,
    my_e_start: Any,
    tile_m: int,
    ep_size: int = 1,
    scoring_fn: Any = "softmax",
    renormalize_topk_logits: Any = False,
) -> Any:
  """Groups tokens into tile_m aligned expert tiles and builds slot_meta."""
  total_tokens, top_k = top_k_indices.shape
  n_assign = total_tokens * top_k
  expected_local_assign = (n_assign + max(1, ep_size) - 1) // max(1, ep_size)
  if scoring_fn == "sigmoid":
    max_local_assign = min(
        n_assign, total_tokens * min(top_k, local_num_experts)
    )
    max_total_tiles = local_num_experts + max(
        0, (max_local_assign - local_num_experts) // tile_m
    )
  elif (
      ep_size > 1
      and local_num_experts >= 8
      and (expected_local_assign <= local_num_experts * tile_m)
  ):
    max_total_tiles = local_num_experts
  else:
    max_local_assign = min(n_assign, expected_local_assign * 2)
    max_total_tiles = local_num_experts + max(
        0, (max_local_assign - local_num_experts) // tile_m
    )
  local_e_2d = top_k_indices - jnp.int32(my_e_start)
  is_local_2d = (local_e_2d >= jnp.int32(0)) & (
      local_e_2d < jnp.int32(local_num_experts)
  )
  safe_local_e_flat = jnp.where(
      is_local_2d, local_e_2d, jnp.int32(local_num_experts)
  ).reshape(-1)
  tok_idx_flat = jax.lax.broadcasted_iota(
      jnp.int32, top_k_indices.shape, 0
  ).reshape(-1)
  weights_flat = top_k_weights.reshape(-1).astype(jnp.float32)
  sort_key = safe_local_e_flat << 16 | tok_idx_flat
  if not renormalize_topk_logits and total_tokens <= 8192:
    weights_s32 = jax.lax.bitcast_convert_type(weights_flat, jnp.int32)
    packed_val = weights_s32 & jnp.int32(-65536) | tok_idx_flat & jnp.int32(
        65535
    )
    _, sorted_val = jax.lax.sort_key_val(sort_key, packed_val)
    sorted_key = None
    sorted_weights = None
  else:
    sorted_val = None
    sorted_key, sorted_weights = jax.lax.sort_key_val(sort_key, weights_flat)
  expert_ids = jnp.arange(local_num_experts, dtype=jnp.int32)
  expert_counts = jnp.sum(
      safe_local_e_flat[:, None] == expert_ids[None, :], axis=0, dtype=jnp.int32
  )
  num_tiles = (expert_counts + jnp.int32(tile_m - 1)) // jnp.int32(tile_m)
  excl_mask = expert_ids[None, :] < expert_ids[:, None]
  excl_packed = jnp.sum(
      jnp.where(
          excl_mask,
          num_tiles[None, :] << 16 | expert_counts[None, :] & jnp.int32(65535),
          jnp.int32(0),
      ),
      axis=1,
      dtype=jnp.int32,
  )
  excl_cum_tiles = excl_packed >> 16
  excl_cum_counts = excl_packed & jnp.int32(65535)
  incl_cum_tiles = excl_cum_tiles + num_tiles
  num_active_tiles = jnp.minimum(
      incl_cum_tiles[-1:] & jnp.int32(2147483647), jnp.int32(max_total_tiles)
  )
  slot_ids = jnp.arange(max_total_tiles, dtype=jnp.int32)
  if local_num_experts > 1:
    ge_mask = (slot_ids[:, None] >= incl_cum_tiles[None, :]).astype(jnp.int32)
    gt_mask = (slot_ids[:, None] >= excl_cum_tiles[None, :]).astype(jnp.int32)
    tile_expert = jnp.minimum(
        jnp.sum(ge_mask, axis=1, dtype=jnp.int32),
        jnp.int32(local_num_experts - 1),
    )
    delta_pad = num_tiles * jnp.int32(tile_m) - expert_counts
    slot_pad_offset = jnp.sum(
        ge_mask * delta_pad[None, :], axis=1, dtype=jnp.int32
    )
    raw_start_s = slot_ids * jnp.int32(tile_m) - slot_pad_offset
    slot_end_s = jnp.sum(
        gt_mask * expert_counts[None, :], axis=1, dtype=jnp.int32
    )
    del excl_cum_counts
  else:
    tile_expert = jnp.zeros((max_total_tiles,), dtype=jnp.int32)
    raw_start_s = slot_ids * jnp.int32(tile_m)
    slot_end_s = jnp.broadcast_to(expert_counts[:1], (max_total_tiles,))
    del excl_cum_counts
  row_in_tile = jnp.arange(tile_m, dtype=jnp.int32)[None, :]
  raw_pos_2d = raw_start_s[:, None] + row_in_tile
  is_valid_row = (
      (slot_ids[:, None] < num_active_tiles)
      & (raw_pos_2d < slot_end_s[:, None])
  ).reshape(-1)
  sorted_pos = jax.lax.clamp(
      jnp.int32(0), raw_pos_2d.reshape(-1), jnp.int32(max(0, n_assign - 1))
  )
  if sorted_val is not None:
    picked_val = sorted_val[sorted_pos]
    src_tok_idx = picked_val & jnp.int32(65535)
    gathered_weights = jax.lax.bitcast_convert_type(
        picked_val & jnp.int32(-65536), jnp.float32
    )
  else:
    assert sorted_key is not None and sorted_weights is not None
    picked_key = sorted_key[sorted_pos]
    gathered_weights = sorted_weights[sorted_pos]
    src_tok_idx = picked_key & jnp.int32(65535)
  padded_tokens = _lock_compact_layout(tokens[src_tok_idx])
  row_weight = jnp.where(is_valid_row, gathered_weights, jnp.float32(0.0))
  num_meta_words = (max_total_tiles + 4 - 1) // 4
  pad_tiles = num_meta_words * 4 - max_total_tiles
  tile_expert_pad = (
      jnp.pad(tile_expert, (0, pad_tiles)) if pad_tiles > 0 else tile_expert
  )
  te_4 = tile_expert_pad.reshape(num_meta_words, 4) & jnp.int32(255)
  packed_words = (
      te_4[:, 0] | te_4[:, 1] << 8 | te_4[:, 2] << 16 | te_4[:, 3] << 24
  )
  slot_meta = jnp.concatenate([packed_words, num_active_tiles], axis=0)
  return (padded_tokens, slot_meta, src_tok_idx, row_weight)


def _dot_2d_f32(lhs: Any, rhs: Any) -> Any:
  """Computes 2D matrix multiplication accumulating in float32 across dtypes."""
  res = jax.lax.dot_general(
      lhs,
      rhs,
      dimension_numbers=(((1,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  return res


def fused_moe_tile_pipeline(
    padded_tokens: Any,
    w1: Any,
    w2: Any,
    slot_meta: Any,
    out: Any,
    w1_scale: Any = None,
    w2_scale: Any = None,
    b1: Any = None,
    b2: Any = None,
    *,
    tile_m: int = 32,
    bf: int = 1024,
    pallas_bf: Any = None,
    pallas_bd1: Any = None,
    pallas_bfc: Any = None,
    vmem_kib: int = 16384,
    act_fn: Any = "silu",
    subc_quant_w1_sz: Any = None,
    subc_quant_w2_sz: Any = None,
    is_tpu7x: Any = True,
) -> Any:
  """Stage 1 jax_k pipelined MoE kernel over active expert token tiles."""
  _, hidden_size = padded_tokens.shape
  _, intermediate_size, _ = w2.shape
  out_dtype = padded_tokens.dtype
  num_bf = intermediate_size // bf
  num_meta_words = slot_meta.shape[0] - 1
  slot_meta = _jax_k_color(slot_meta, "vmem", is_output=True)
  meta_words = [
      _read_smem_scalar(slot_meta, idx) for idx in range(num_meta_words)
  ]
  total_steps = _read_smem_scalar(slot_meta, num_meta_words)
  padded_tokens = _pin_to_hbm(padded_tokens)
  if configs.itemsize_bits(w1.dtype) == 4:
    w1 = _pin_to_hbm(w1.astype(jnp.float8_e5m2))
    w2 = _pin_to_hbm(w2.astype(jnp.float8_e5m2))
  else:
    w1 = _lock_compact_layout(w1)
    w2 = _lock_compact_layout(w2)
  out = _pin_to_hbm(out)
  scale_m = 8 if tile_m > 32 else tile_m
  raw_subc_1 = 0
  is_per_channel_w1 = False
  w1_scale_0_3d = None
  w3_scale_1_3d = None
  if w1_scale is not None:
    assert subc_quant_w1_sz is not None
    raw_subc_1 = hidden_size // subc_quant_w1_sz
    is_per_channel_w1 = raw_subc_1 == 1
    if is_per_channel_w1:
      num_e_local = w1_scale.shape[0]
      w1_scale_0_3d = _lock_compact_layout(
          jax.lax.broadcast_in_dim(
              jnp.reshape(
                  w1_scale[:, 0:1, 0, 0, :], (num_e_local, 1, intermediate_size)
              ),
              (num_e_local, scale_m, intermediate_size),
              (0, 1, 2),
          )
      )
      w3_scale_1_3d = _lock_compact_layout(
          jax.lax.broadcast_in_dim(
              jnp.reshape(
                  w1_scale[:, 1:2, 0, 0, :], (num_e_local, 1, intermediate_size)
              ),
              (num_e_local, scale_m, intermediate_size),
              (0, 1, 2),
          )
      )
    else:
      w1_scale = _lock_compact_layout(w1_scale)
  is_per_channel_w2 = False
  w2_scale_0_3d = None
  if w2_scale is not None:
    assert subc_quant_w2_sz is not None
    is_per_channel_w2 = subc_quant_w2_sz >= intermediate_size
    if is_per_channel_w2:
      num_e_local = w2_scale.shape[0]
      w2_scale_0_3d = _lock_compact_layout(
          jax.lax.broadcast_in_dim(
              jnp.reshape(w2_scale[:, 0, 0, :], (num_e_local, 1, hidden_size)),
              (num_e_local, scale_m, hidden_size),
              (0, 1, 2),
          )
      )
    else:
      w2_scale = _lock_compact_layout(w2_scale)
  if is_per_channel_w2 and pallas_bf is not None and (pallas_bf > 0):
    bf = math.gcd(bf, pallas_bf)
    num_bf = intermediate_size // bf
  if b1 is not None:
    b1 = _lock_compact_layout(b1)
  if b2 is not None:
    b2 = _lock_compact_layout(b2)
  use_sub_bf_trunc = is_per_channel_w2 or hidden_size >= 4096
  use_split_h = is_per_channel_w1 and hidden_size >= 4096 and (num_bf > 1)
  hoist_ffn1 = (
      w1_scale is None and num_bf > 2 or (is_per_channel_w1 and num_bf > 1)
  )
  ffn1_bf = intermediate_size if hoist_ffn1 else bf
  h_half = hidden_size // 2
  del pallas_bf, pallas_bd1, pallas_bfc, is_tpu7x
  eff_w2_sub_sz = 0
  raw_subc_2 = 0
  if (
      not is_per_channel_w2
      and w2_scale is not None
      and (subc_quant_w2_sz is not None)
  ):
    eff_w2_sub_sz = min(bf, subc_quant_w2_sz)
    raw_subc_2 = bf // eff_w2_sub_sz
  is_large_sharded = hidden_size >= 3072 and w1.shape[0] <= 80
  scoped_vmem_kib = 36864 if is_large_sharded else None
  use_must_fuse_up_down = (
      (hidden_size < 3072 or is_large_sharded)
      and (not use_split_h)
      and (act_fn != "gelu")
      and (tile_m <= 32 or hidden_size <= 2048)
      and (w1_scale is None or is_per_channel_w1)
      and (w2_scale is None or is_per_channel_w2)
  )

  def _expand_scale_tile(scale_in: Any, cols: int) -> Any:
    if scale_m == tile_m:
      return jnp.reshape(scale_in, (tile_m, cols))
    return jax.lax.broadcast_in_dim(
        jnp.reshape(scale_in, (1, scale_m, cols)),
        (tile_m // scale_m, scale_m, cols),
        (0, 1, 2),
    ).reshape(tile_m, cols)

  def _expand_bias_tile(bias_in: Any, cols: int) -> Any:
    if bias_in is None:
      return None
    return jax.lax.broadcast_in_dim(
        jnp.reshape(bias_in, (1, cols)), (tile_m, cols), (0, 1)
    )

  k_ffn1 = h_half if use_split_h else hidden_size
  ffn1_n_win = ffn1_bf // 2
  ffn1_iwb = (tile_m // 8, k_ffn1 // 128) if hoist_ffn1 else None
  ffn1_owb = (tile_m // 8, ffn1_n_win // 128) if hoist_ffn1 else None
  ffn1_kwb = (k_ffn1 // 8, ffn1_n_win // 128) if hoist_ffn1 else None
  down_n_win = hidden_size // 2 if hidden_size >= 4096 else hidden_size
  down_iwb = (tile_m // 8, bf // 128)
  down_owb = (tile_m // 8, down_n_win // 128)
  down_kwb = (bf // 8, down_n_win // 128)

  @_jax_k_fusion(
      vmem_kib=vmem_kib,
      scoped_vmem_limit_kib=scoped_vmem_kib,
      buffering_level=2,
      must_fuse=True,
      input_window_bounds=ffn1_iwb,
      output_window_bounds=ffn1_owb,
      kernel_window_bounds=ffn1_kwb,
  )
  def _ffn1_split_h_tile_fusion(
      x_0_in: Any,
      x_1_in: Any,
      w1_0_2d: Any,
      w1_1_2d: Any,
      w3_0_2d: Any,
      w3_1_2d: Any,
      s1_tile: Any,
      s3_tile: Any,
      b1_tile: Any,
      b3_tile: Any,
  ) -> Any:
    acc1_0 = jax.lax.clamp(
        jnp.float32(-1e30),
        _dot_2d_f32(x_0_in, w1_0_2d) * s1_tile,
        jnp.float32(1e30),
    )
    acc1 = acc1_0 + _dot_2d_f32(x_1_in, w1_1_2d) * s1_tile
    if b1_tile is not None:
      acc1 = acc1 + b1_tile
    gate = _apply_gate_fn(acc1, act_fn)
    acc3_0 = jax.lax.clamp(
        jnp.float32(-1e30),
        _dot_2d_f32(x_0_in, w3_0_2d) * s3_tile,
        jnp.float32(1e30),
    )
    acc3 = acc3_0 + _dot_2d_f32(x_1_in, w3_1_2d) * s3_tile
    if b3_tile is not None:
      acc3 = acc3 + b3_tile
    act = _combine_gate_up(gate, acc3, act_fn)
    if out_dtype == jnp.bfloat16:
      act = act.astype(jnp.bfloat16)
    return _lock_compact_layout(act)

  @_jax_k_fusion(
      vmem_kib=vmem_kib,
      scoped_vmem_limit_kib=scoped_vmem_kib,
      buffering_level=2,
      input_window_bounds=ffn1_iwb,
      output_window_bounds=ffn1_owb,
      kernel_window_bounds=ffn1_kwb,
  )
  def _gate_tile_fusion(
      x_0_in: Any, w1_0_in: Any, w1_scale_0_in: Any, b1_0_in: Any
  ) -> Any:
    w1_2d = jnp.reshape(w1_0_in, (hidden_size, ffn1_bf))
    if w1_scale_0_in is None:
      acc1 = _dot_2d_f32(x_0_in, w1_2d)
    elif is_per_channel_w1:
      s1_tile = _expand_scale_tile(w1_scale_0_in, ffn1_bf)
      acc1 = _dot_2d_f32(x_0_in, w1_2d) * s1_tile
    else:
      assert subc_quant_w1_sz is not None
      s1_2d = jax.lax.broadcast_in_dim(
          jnp.reshape(
              w1_scale_0_in.astype(out_dtype), (raw_subc_1, 1, ffn1_bf)
          ),
          (raw_subc_1, subc_quant_w1_sz, ffn1_bf),
          (0, 1, 2),
      ).reshape(hidden_size, ffn1_bf)
      w1_deq = w1_2d.astype(out_dtype) * s1_2d
      acc1 = _dot_2d_f32(x_0_in, w1_deq)
    if b1_0_in is not None:
      acc1 = acc1 + jax.lax.broadcast_in_dim(
          jnp.reshape(b1_0_in, (1, ffn1_bf)), (tile_m, ffn1_bf), (0, 1)
      )
    gate = _apply_gate_fn(acc1, act_fn)
    return _lock_compact_layout(gate)

  @_jax_k_fusion(
      vmem_kib=vmem_kib,
      scoped_vmem_limit_kib=scoped_vmem_kib,
      buffering_level=2,
      input_window_bounds=ffn1_iwb,
      output_window_bounds=ffn1_owb,
      kernel_window_bounds=ffn1_kwb,
  )
  def _up_tile_fusion(
      gate_in: Any, x_0_in: Any, w3_0_in: Any, w3_scale_1_in: Any, b3_1_in: Any
  ) -> Any:
    w3_2d = jnp.reshape(w3_0_in, (hidden_size, ffn1_bf))
    if w3_scale_1_in is None:
      acc3 = _dot_2d_f32(x_0_in, w3_2d)
    elif is_per_channel_w1:
      s3_tile = _expand_scale_tile(w3_scale_1_in, ffn1_bf)
      acc3 = _dot_2d_f32(x_0_in, w3_2d) * s3_tile
    else:
      assert subc_quant_w1_sz is not None
      s3_2d = jax.lax.broadcast_in_dim(
          jnp.reshape(
              w3_scale_1_in.astype(out_dtype), (raw_subc_1, 1, ffn1_bf)
          ),
          (raw_subc_1, subc_quant_w1_sz, ffn1_bf),
          (0, 1, 2),
      ).reshape(hidden_size, ffn1_bf)
      w3_deq = w3_2d.astype(out_dtype) * s3_2d
      acc3 = _dot_2d_f32(x_0_in, w3_deq)
    if b3_1_in is not None:
      acc3 = acc3 + jax.lax.broadcast_in_dim(
          jnp.reshape(b3_1_in, (1, ffn1_bf)), (tile_m, ffn1_bf), (0, 1)
      )
    act = _combine_gate_up(gate_in, acc3, act_fn)
    if out_dtype == jnp.bfloat16:
      act = act.astype(jnp.bfloat16)
    return _lock_compact_layout(act)

  @_jax_k_fusion(
      vmem_kib=vmem_kib,
      scoped_vmem_limit_kib=scoped_vmem_kib,
      buffering_level=2,
      input_window_bounds=down_iwb,
      output_window_bounds=down_owb,
      kernel_window_bounds=down_kwb,
  )
  def _down_tile_fusion(
      act_in: Any, w2_in: Any, w2_scale_in: Any, b2_in: Any, accum_in: Any
  ) -> Any:
    w2_2d = jnp.reshape(w2_in, (bf, hidden_size))
    if w2_scale_in is None:
      res_sub = _dot_2d_f32(act_in, w2_2d)
    elif is_per_channel_w2:
      s2_tile = _expand_scale_tile(w2_scale_in, hidden_size)
      res_sub = _dot_2d_f32(act_in, w2_2d) * s2_tile
    else:
      s2_2d = jax.lax.broadcast_in_dim(
          jnp.reshape(
              w2_scale_in.astype(out_dtype), (raw_subc_2, 1, hidden_size)
          ),
          (raw_subc_2, eff_w2_sub_sz, hidden_size),
          (0, 1, 2),
      ).reshape(bf, hidden_size)
      w2_deq = w2_2d.astype(out_dtype) * s2_2d
      res_sub = _dot_2d_f32(act_in, w2_deq)
    if b2_in is not None:
      res_sub = res_sub + jax.lax.broadcast_in_dim(
          jnp.reshape(b2_in, (1, hidden_size)), (tile_m, hidden_size), (0, 1)
      )
    if use_sub_bf_trunc and out_dtype == jnp.bfloat16:
      res_sub = jax.lax.bitcast_convert_type(
          jax.lax.bitcast_convert_type(res_sub, jnp.uint32)
          & jnp.uint32(4294901760),
          jnp.float32,
      )
    res_out = res_sub.astype(out_dtype)
    if accum_in is not None:
      res_out = accum_in + res_out
    return _lock_compact_layout(res_out)

  def _prepare_up_down_must_fuse_operands(
      w1_0_in: Any,
      w3_0_in: Any,
      w2_in: Any,
      w1_scale_0_in: Any,
      w3_scale_1_in: Any,
      w2_scale_in: Any,
      b1_0_in: Any,
      b3_1_in: Any,
      b2_in: Any,
  ) -> Any:
    w2_2d = jnp.reshape(w2_in, (bf, hidden_size))
    w1_2d = jnp.reshape(w1_0_in, (hidden_size, bf))
    w3_2d = jnp.reshape(w3_0_in, (hidden_size, bf))
    s1_tile = (
        _expand_scale_tile(w1_scale_0_in, bf)
        if w1_scale_0_in is not None
        else None
    )
    s3_tile = (
        _expand_scale_tile(w3_scale_1_in, bf)
        if w3_scale_1_in is not None
        else None
    )
    s2_tile = (
        _expand_scale_tile(w2_scale_in, hidden_size)
        if w2_scale_in is not None
        else None
    )
    b1_tile = _expand_bias_tile(b1_0_in, bf)
    b3_tile = _expand_bias_tile(b3_1_in, bf)
    b2_tile = _expand_bias_tile(b2_in, hidden_size)
    return (
        w1_2d,
        w3_2d,
        w2_2d,
        s1_tile,
        s3_tile,
        s2_tile,
        b1_tile,
        b3_tile,
        b2_tile,
    )

  @_jax_k_fusion(
      vmem_kib=vmem_kib,
      scoped_vmem_limit_kib=scoped_vmem_kib,
      buffering_level=2,
      must_fuse=use_must_fuse_up_down,
  )
  def _up_down_tile_fusion(
      x_0_in: Any,
      x_1_in: Any,
      w1_0_in: Any,
      w1_1_in: Any,
      w3_0_in: Any,
      w3_1_in: Any,
      w2_in: Any,
      w1_scale_0_in: Any,
      w3_scale_1_in: Any,
      w2_scale_in: Any,
      b1_0_in: Any,
      b3_1_in: Any,
      b2_in: Any,
  ) -> Any:
    if use_must_fuse_up_down:
      w2_2d = w2_in
      w1_2d = w1_0_in
      w3_2d = w3_0_in
      w1_0_2d = None
      w1_1_2d = None
      w3_0_2d = None
      w3_1_2d = None
      w1_deq = None
      w3_deq = None
      w2_deq = None
      s1_tile = w1_scale_0_in
      s3_tile = w3_scale_1_in
      s2_tile = w2_scale_in
      b1_tile = b1_0_in
      b3_tile = b3_1_in
      b2_tile = b2_in
    else:
      w2_2d = jnp.reshape(w2_in, (bf, hidden_size))
      if use_split_h:
        assert w1_1_in is not None and w3_1_in is not None
        w1_0_2d = jnp.reshape(w1_0_in, (h_half, bf))
        w1_1_2d = jnp.reshape(w1_1_in, (h_half, bf))
        w3_0_2d = jnp.reshape(w3_0_in, (h_half, bf))
        w3_1_2d = jnp.reshape(w3_1_in, (h_half, bf))
        w1_2d = None
        w3_2d = None
      else:
        w1_0_2d = None
        w1_1_2d = None
        w3_0_2d = None
        w3_1_2d = None
        w1_2d = jnp.reshape(w1_0_in, (hidden_size, bf))
        w3_2d = jnp.reshape(w3_0_in, (hidden_size, bf))
      w1_deq = None
      if w1_scale_0_in is not None and (not is_per_channel_w1):
        assert subc_quant_w1_sz is not None and w1_2d is not None
        s1_2d = jax.lax.broadcast_in_dim(
            jnp.reshape(w1_scale_0_in.astype(out_dtype), (raw_subc_1, 1, bf)),
            (raw_subc_1, subc_quant_w1_sz, bf),
            (0, 1, 2),
        ).reshape(hidden_size, bf)
        w1_deq = w1_2d.astype(out_dtype) * s1_2d
      w3_deq = None
      if w3_scale_1_in is not None and (not is_per_channel_w1):
        assert subc_quant_w1_sz is not None and w3_2d is not None
        s3_2d = jax.lax.broadcast_in_dim(
            jnp.reshape(w3_scale_1_in.astype(out_dtype), (raw_subc_1, 1, bf)),
            (raw_subc_1, subc_quant_w1_sz, bf),
            (0, 1, 2),
        ).reshape(hidden_size, bf)
        w3_deq = w3_2d.astype(out_dtype) * s3_2d
      w2_deq = None
      if w2_scale_in is not None and (not is_per_channel_w2):
        s2_2d = jax.lax.broadcast_in_dim(
            jnp.reshape(
                w2_scale_in.astype(out_dtype), (raw_subc_2, 1, hidden_size)
            ),
            (raw_subc_2, eff_w2_sub_sz, hidden_size),
            (0, 1, 2),
        ).reshape(bf, hidden_size)
        w2_deq = w2_2d.astype(out_dtype) * s2_2d
      s1_tile = (
          _expand_scale_tile(w1_scale_0_in, bf)
          if w1_scale_0_in is not None and is_per_channel_w1
          else None
      )
      s3_tile = (
          _expand_scale_tile(w3_scale_1_in, bf)
          if w3_scale_1_in is not None and is_per_channel_w1
          else None
      )
      s2_tile = (
          _expand_scale_tile(w2_scale_in, hidden_size)
          if w2_scale_in is not None and is_per_channel_w2
          else None
      )
      b1_tile = (
          jax.lax.broadcast_in_dim(
              jnp.reshape(b1_0_in, (1, bf)), (tile_m, bf), (0, 1)
          )
          if b1_0_in is not None
          else None
      )
      b3_tile = (
          jax.lax.broadcast_in_dim(
              jnp.reshape(b3_1_in, (1, bf)), (tile_m, bf), (0, 1)
          )
          if b3_1_in is not None
          else None
      )
      b2_tile = (
          jax.lax.broadcast_in_dim(
              jnp.reshape(b2_in, (1, hidden_size)),
              (tile_m, hidden_size),
              (0, 1),
          )
          if b2_in is not None
          else None
      )
    if use_split_h:
      assert (
          x_1_in is not None
          and w1_0_2d is not None
          and (w1_1_2d is not None)
          and (s1_tile is not None)
      )
      acc1_0 = jax.lax.clamp(
          jnp.float32(-1e30),
          _dot_2d_f32(x_0_in, w1_0_2d) * s1_tile,
          jnp.float32(1e30),
      )
      acc1 = acc1_0 + _dot_2d_f32(x_1_in, w1_1_2d) * s1_tile
    elif w1_scale_0_in is None:
      assert w1_2d is not None
      acc1 = _dot_2d_f32(x_0_in, w1_2d)
    elif is_per_channel_w1:
      assert w1_2d is not None and s1_tile is not None
      acc1 = _dot_2d_f32(x_0_in, w1_2d) * s1_tile
    else:
      assert w1_deq is not None
      acc1 = _dot_2d_f32(x_0_in, w1_deq)
    if b1_tile is not None:
      acc1 = acc1 + b1_tile
    gate_in = _apply_gate_fn(acc1, act_fn)
    if use_split_h:
      assert (
          x_1_in is not None
          and w3_0_2d is not None
          and (w3_1_2d is not None)
          and (s3_tile is not None)
      )
      acc3_0 = jax.lax.clamp(
          jnp.float32(-1e30),
          _dot_2d_f32(x_0_in, w3_0_2d) * s3_tile,
          jnp.float32(1e30),
      )
      acc3 = acc3_0 + _dot_2d_f32(x_1_in, w3_1_2d) * s3_tile
    elif w3_scale_1_in is None:
      assert w3_2d is not None
      acc3 = _dot_2d_f32(x_0_in, w3_2d)
    elif is_per_channel_w1:
      assert w3_2d is not None and s3_tile is not None
      acc3 = _dot_2d_f32(x_0_in, w3_2d) * s3_tile
    else:
      assert w3_deq is not None
      acc3 = _dot_2d_f32(x_0_in, w3_deq)
    if b3_tile is not None:
      acc3 = acc3 + b3_tile
    act = _combine_gate_up(gate_in, acc3, act_fn)
    if out_dtype == jnp.bfloat16:
      act = act.astype(jnp.bfloat16)
    if w2_scale_in is None:
      res_sub = _dot_2d_f32(act, w2_2d)
    elif is_per_channel_w2:
      assert s2_tile is not None
      res_sub = _dot_2d_f32(act, w2_2d) * s2_tile
    else:
      assert w2_deq is not None
      res_sub = _dot_2d_f32(act, w2_deq)
    if b2_tile is not None:
      res_sub = res_sub + b2_tile
    if use_sub_bf_trunc and out_dtype == jnp.bfloat16:
      res_sub = jax.lax.bitcast_convert_type(
          jax.lax.bitcast_convert_type(res_sub, jnp.uint32)
          & jnp.uint32(4294901760),
          jnp.float32,
      )
    res_out = res_sub.astype(out_dtype)
    return _lock_compact_layout(res_out)

  def step_body(s: int, acc_out: Any) -> Any:
    acc_out = _lock_compact_layout(acc_out)
    padded_tokens_c = _lock_compact_layout(padded_tokens)
    w1_c = _lock_compact_layout(w1)
    w2_c = _lock_compact_layout(w2)
    w1_scale_c = _lock_compact_layout(w1_scale)
    w1_scale_0_3d_c = _lock_compact_layout(w1_scale_0_3d)
    w3_scale_1_3d_c = _lock_compact_layout(w3_scale_1_3d)
    w2_scale_c = _lock_compact_layout(w2_scale)
    w2_scale_0_3d_c = _lock_compact_layout(w2_scale_0_3d)
    b1_c = _lock_compact_layout(b1)
    b2_c = _lock_compact_layout(b2)
    word_idx = s >> jnp.int32(2)
    word = _select_word_tree(meta_words, word_idx, 0, num_meta_words)
    shift_bits = (s & jnp.int32(3)) << jnp.int32(3)
    e = word >> shift_bits & jnp.int32(255)
    row_offset = s * jnp.int32(tile_m)
    if use_split_h:
      x_0_tile = _jax_k_block_slice(
          padded_tokens_c,
          (
              jnp.asarray(row_offset, dtype=jnp.int32),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
          ),
          (tile_m, h_half),
      )
      x_0_vmem = x_0_tile
      x_1_tile = _jax_k_block_slice(
          padded_tokens_c,
          (
              jnp.asarray(row_offset, dtype=jnp.int32),
              jnp.asarray(jnp.int32(h_half), dtype=jnp.int32),
          ),
          (tile_m, h_half),
      )
      x_1_vmem = x_1_tile
    else:
      x_0_tile = _jax_k_block_slice(
          padded_tokens_c,
          (
              jnp.asarray(row_offset, dtype=jnp.int32),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
          ),
          (tile_m, hidden_size),
      )
      x_0_vmem = x_0_tile
      x_1_vmem = None
    act_full = None
    if hoist_ffn1:
      f0 = jnp.int32(0)
      if use_split_h:
        w1_0_slice = _jax_k_block_slice(
            w1_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, h_half, ffn1_bf),
        )
        w1_0_vmem = w1_0_slice
        w1_1_slice = _jax_k_block_slice(
            w1_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(h_half), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, h_half, ffn1_bf),
        )
        w1_1_vmem = w1_1_slice
        w3_0_slice = _jax_k_block_slice(
            w1_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, h_half, ffn1_bf),
        )
        w3_0_vmem = w3_0_slice
        w3_1_slice = _jax_k_block_slice(
            w1_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                jnp.asarray(jnp.int32(h_half), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, h_half, ffn1_bf),
        )
        w3_1_vmem = w3_1_slice
      else:
        w1_0_slice = _jax_k_block_slice(
            w1_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, hidden_size, ffn1_bf),
        )
        w1_0_vmem = w1_0_slice
        w1_1_vmem = None
        w3_0_slice = _jax_k_block_slice(
            w1_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, hidden_size, ffn1_bf),
        )
        w3_0_vmem = w3_0_slice
        w3_1_vmem = None
      w1_scale_0_vmem = None
      w3_scale_1_vmem = None
      if w1_scale_0_3d_c is not None and w3_scale_1_3d_c is not None:
        w1_scale_0_slice = _jax_k_block_slice(
            w1_scale_0_3d_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, scale_m, ffn1_bf),
        )
        w1_scale_0_vmem = w1_scale_0_slice
        w3_scale_1_slice = _jax_k_block_slice(
            w3_scale_1_3d_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, scale_m, ffn1_bf),
        )
        w3_scale_1_vmem = w3_scale_1_slice
      elif w1_scale_c is not None:
        w1_scale_0_slice = _jax_k_block_slice(
            w1_scale_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, raw_subc_1, 1, ffn1_bf),
        )
        w1_scale_0_vmem = w1_scale_0_slice
        w3_scale_1_slice = _jax_k_block_slice(
            w1_scale_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, raw_subc_1, 1, ffn1_bf),
        )
        w3_scale_1_vmem = w3_scale_1_slice
      b1_0_vmem = None
      b3_1_vmem = None
      if b1_c is not None:
        b1_0_slice = _jax_k_block_slice(
            b1_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, 1, ffn1_bf),
        )
        b1_0_vmem = b1_0_slice
        b3_1_slice = _jax_k_block_slice(
            b1_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(f0, dtype=jnp.int32),
            ),
            (1, 1, 1, ffn1_bf),
        )
        b3_1_vmem = b3_1_slice
      if use_split_h:
        assert (
            x_1_vmem is not None
            and w1_1_vmem is not None
            and (w3_1_vmem is not None)
            and (w1_scale_0_vmem is not None)
            and (w3_scale_1_vmem is not None)
        )
        w1_0_2d = jnp.reshape(w1_0_vmem, (h_half, ffn1_bf))
        w1_1_2d = jnp.reshape(w1_1_vmem, (h_half, ffn1_bf))
        w3_0_2d = jnp.reshape(w3_0_vmem, (h_half, ffn1_bf))
        w3_1_2d = jnp.reshape(w3_1_vmem, (h_half, ffn1_bf))
        s1_tile = _expand_scale_tile(w1_scale_0_vmem, ffn1_bf)
        s3_tile = _expand_scale_tile(w3_scale_1_vmem, ffn1_bf)
        b1_tile = _expand_bias_tile(b1_0_vmem, ffn1_bf)
        b3_tile = _expand_bias_tile(b3_1_vmem, ffn1_bf)
        act_full = _ffn1_split_h_tile_fusion(
            x_0_vmem,
            x_1_vmem,
            w1_0_2d,
            w1_1_2d,
            w3_0_2d,
            w3_1_2d,
            s1_tile,
            s3_tile,
            b1_tile,
            b3_tile,
        )
      else:
        gate_full = _gate_tile_fusion(
            x_0_vmem, w1_0_vmem, w1_scale_0_vmem, b1_0_vmem
        )
        act_full = _up_tile_fusion(
            gate_full, x_0_vmem, w3_0_vmem, w3_scale_1_vmem, b3_1_vmem
        )
      act_full = _lock_compact_layout(act_full)
    out_tile_accum = None
    for bf_id in range(num_bf):
      f_offset = jnp.int32(bf_id * bf)
      w1_0_vmem = None
      w1_1_vmem = None
      w1_scale_0_vmem = None
      b1_0_vmem = None
      w3_0_vmem = None
      w3_1_vmem = None
      w3_scale_1_vmem = None
      b3_1_vmem = None
      if hoist_ffn1:
        assert act_full is not None
        if num_bf == 1:
          act_tile = act_full
        else:
          act_tile = _lock_compact_layout(
              jax.lax.slice_in_dim(
                  act_full, bf_id * bf, (bf_id + 1) * bf, axis=1
              )
          )
      else:
        if use_split_h:
          w1_0_slice = _jax_k_block_slice(
              w1_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, h_half, bf),
          )
          w1_0_vmem = w1_0_slice
          w1_1_slice = _jax_k_block_slice(
              w1_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(h_half), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, h_half, bf),
          )
          w1_1_vmem = w1_1_slice
          w3_0_slice = _jax_k_block_slice(
              w1_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, h_half, bf),
          )
          w3_0_vmem = w3_0_slice
          w3_1_slice = _jax_k_block_slice(
              w1_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(h_half), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, h_half, bf),
          )
          w3_1_vmem = w3_1_slice
        else:
          w1_0_slice = _jax_k_block_slice(
              w1_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, hidden_size, bf),
          )
          w1_0_vmem = w1_0_slice
          w1_1_vmem = None
          w3_0_slice = _jax_k_block_slice(
              w1_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, hidden_size, bf),
          )
          w3_0_vmem = w3_0_slice
          w3_1_vmem = None
        w1_scale_0_vmem = None
        w3_scale_1_vmem = None
        if w1_scale_0_3d_c is not None and w3_scale_1_3d_c is not None:
          w1_scale_0_slice = _jax_k_block_slice(
              w1_scale_0_3d_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, scale_m, bf),
          )
          w1_scale_0_vmem = w1_scale_0_slice
          w3_scale_1_slice = _jax_k_block_slice(
              w3_scale_1_3d_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, scale_m, bf),
          )
          w3_scale_1_vmem = w3_scale_1_slice
        elif w1_scale_c is not None:
          w1_scale_0_slice = _jax_k_block_slice(
              w1_scale_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, raw_subc_1, 1, bf),
          )
          w1_scale_0_vmem = w1_scale_0_slice
          w3_scale_1_slice = _jax_k_block_slice(
              w1_scale_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, raw_subc_1, 1, bf),
          )
          w3_scale_1_vmem = w3_scale_1_slice
        b1_0_vmem = None
        b3_1_vmem = None
        if b1_c is not None:
          b1_0_slice = _jax_k_block_slice(
              b1_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, 1, bf),
          )
          b1_0_vmem = b1_0_slice
          b3_1_slice = _jax_k_block_slice(
              b1_c,
              (
                  jnp.asarray(e, dtype=jnp.int32),
                  jnp.asarray(jnp.int32(1), dtype=jnp.int32),
                  jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                  jnp.asarray(f_offset, dtype=jnp.int32),
              ),
              (1, 1, 1, bf),
          )
          b3_1_vmem = b3_1_slice
        act_tile = None
      w2_slice = _jax_k_block_slice(
          w2_c,
          (
              jnp.asarray(e, dtype=jnp.int32),
              jnp.asarray(f_offset, dtype=jnp.int32),
              jnp.asarray(jnp.int32(0), dtype=jnp.int32),
          ),
          (1, bf, hidden_size),
      )
      w2_vmem = w2_slice
      w2_scale_vmem = None
      if is_per_channel_w2 and w2_scale_0_3d_c is not None:
        w2_scale_0_slice = _jax_k_block_slice(
            w2_scale_0_3d_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
            ),
            (1, scale_m, hidden_size),
        )
        w2_scale_vmem = w2_scale_0_slice
      elif w2_scale_c is not None:
        f_scale_offset = jnp.int32(bf_id * bf // eff_w2_sub_sz)
        w2_scale_sub_slice = _jax_k_block_slice(
            w2_scale_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(f_scale_offset, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
            ),
            (1, raw_subc_2, 1, hidden_size),
        )
        w2_scale_vmem = w2_scale_sub_slice
      b2_vmem = None
      if b2_c is not None and bf_id == 0:
        b2_slice = _jax_k_block_slice(
            b2_c,
            (
                jnp.asarray(e, dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
                jnp.asarray(jnp.int32(0), dtype=jnp.int32),
            ),
            (1, 1, hidden_size),
        )
        b2_vmem = b2_slice
      if hoist_ffn1:
        assert act_tile is not None
        out_tile_accum = _down_tile_fusion(
            act_tile, w2_vmem, w2_scale_vmem, b2_vmem, out_tile_accum
        )
        if num_bf > 1 and bf_id != num_bf - 1:
          out_tile_accum = _lock_compact_layout(out_tile_accum)
      else:
        if use_must_fuse_up_down:
          (
              w1_0_vmem,
              w3_0_vmem,
              w2_vmem,
              w1_scale_0_vmem,
              w3_scale_1_vmem,
              w2_scale_vmem,
              b1_0_vmem,
              b3_1_vmem,
              b2_vmem,
          ) = _prepare_up_down_must_fuse_operands(
              w1_0_vmem,
              w3_0_vmem,
              w2_vmem,
              w1_scale_0_vmem,
              w3_scale_1_vmem,
              w2_scale_vmem,
              b1_0_vmem,
              b3_1_vmem,
              b2_vmem,
          )
        part_tile = _up_down_tile_fusion(
            x_0_vmem,
            x_1_vmem,
            w1_0_vmem,
            w1_1_vmem,
            w3_0_vmem,
            w3_1_vmem,
            w2_vmem,
            w1_scale_0_vmem,
            w3_scale_1_vmem,
            w2_scale_vmem,
            b1_0_vmem,
            b3_1_vmem,
            b2_vmem,
        )
        if out_tile_accum is None:
          out_tile_accum = part_tile
        else:
          out_tile_accum = out_tile_accum + part_tile
    assert out_tile_accum is not None
    if not is_large_sharded:
      out_tile_accum = _lock_compact_layout(
          jax.lax.optimization_barrier(out_tile_accum)
      )
      updated = _jax_k_block_update(
          acc_out,
          out_tile_accum,
          (row_offset, jnp.int32(0)),
          in_pipelined_loop=False,
      )
    else:
      with xla_metadata.set_xla_metadata(buffering_level="2"):
        updated = _jax_k_block_update(
            acc_out,
            _lock_compact_layout(out_tile_accum),
            (row_offset, jnp.int32(0)),
            in_pipelined_loop=False,
        )
    return _lock_compact_layout(updated)

  return _jax_k_pipeline(
      lower=0,
      upper=total_steps,
      init=out,
      body=step_body,
      buffering_level=1,
      boundary_mode="auto",
      async_pipelining_levels=0,
      has_dus=True,
  )


@functools.partial(
    jax.jit,
    static_argnames=[
        "mesh",
        "top_k",
        "renormalize_topk_logits",
        "act_fn",
        "scoring_fn",
        "subc_quant_w1_sz",
        "subc_quant_w2_sz",
        "bt",
        "bf",
        "bd1",
        "bd2",
        "btc",
        "bfc",
        "bd1c",
        "bd2c",
        "ep_axis_name",
    ],
)
def fused_ep_moe(
    mesh: Any,
    tokens: Any,
    w1: Any,
    w2: Any,
    gating_output: Any,
    top_k: int,
    *,
    renormalize_topk_logits: Any = False,
    act_fn: Any = "silu",
    scoring_fn: Any = "softmax",
    subc_quant_w1_sz: Any = None,
    subc_quant_w2_sz: Any = None,
    w1_scale: Any = None,
    w2_scale: Any = None,
    b1: Any = None,
    b2: Any = None,
    bt: Any = None,
    bf: Any = None,
    bd1: Any = None,
    bd2: Any = None,
    btc: Any = None,
    bfc: Any = None,
    bd1c: Any = None,
    bd2c: Any = None,
    ep_axis_name: Any = "model",
) -> Any:
  """Stage 1 jax_k Fused Mixture of Experts (MoE) entrypoint."""
  if mesh is None:
    mesh = configs.create_default_mesh(ep_axis_name=ep_axis_name)
  if len(mesh.shape) != 2:
    raise NotImplementedError("Only 2D mesh is supported.")
  for axis_name in mesh.axis_names:
    if axis_name == ep_axis_name:
      continue
    if mesh.shape[axis_name] != 1:
      raise NotImplementedError(
          "Expected all non ep axis to have size 1 in"
          f" mesh.shape={mesh.shape!r}"
      )
  ep_size = mesh.shape[ep_axis_name]
  num_tokens, hidden_size = tokens.shape
  num_experts, intermediate_size, _ = w2.shape
  if w1.shape != (num_experts, 2, hidden_size, intermediate_size):
    raise ValueError(
        f"Expected w1.shape={w1.shape!r} to be"
        f" {(num_experts, 2, hidden_size, intermediate_size)}."
    )
  if w2.shape != (num_experts, intermediate_size, hidden_size):
    raise ValueError(
        f"Expected w2.shape={w2.shape!r} to be"
        f" {(num_experts, intermediate_size, hidden_size)}."
    )
  if gating_output.shape != (num_tokens, num_experts):
    raise ValueError(
        f"Expected gating_output.shape={gating_output.shape!r} to be"
        f" {(num_tokens, num_experts)}."
    )
  if not 0 < top_k <= num_experts:
    raise ValueError(
        f"Expected top_k={top_k!r} to be in range (0,"
        f" num_experts={num_experts!r}]."
    )
  if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
    raise ValueError(
        f"Expected hidden_size={hidden_size!r} and"
        f" intermediate_size={intermediate_size!r} to be aligned to 128."
    )
  if num_tokens % ep_size != 0:
    raise ValueError(
        f"Expected num_tokens={num_tokens!r} to be aligned to"
        f" ep_size={ep_size!r}."
    )
  if num_experts % ep_size != 0:
    raise ValueError(
        f"Expected num_experts={num_experts!r} to be aligned to"
        f" ep_size={ep_size!r}."
    )
  local_num_tokens = num_tokens // ep_size
  local_num_experts = num_experts // ep_size
  padded_num_experts = align_to(num_experts, 128)
  t_dtype = tokens.dtype
  t_packing = get_dtype_packing(t_dtype)
  assert w1.dtype == w2.dtype
  w_packing = get_dtype_packing(w1.dtype)
  if None in {bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c}:
    bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c = get_tuned_block_sizes(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        t_packing=t_packing,
        w_packing=w_packing,
        num_tokens=num_tokens,
        ep_size=ep_size,
    )
  if local_num_tokens <= t_packing * 8:
    bt = local_num_tokens
    btc = bt
  assert bt is not None and btc is not None
  bt = min(local_num_tokens, bt)
  btc = min(bt, btc)
  assert bt % btc == 0
  if local_num_tokens % t_packing != 0:
    raise ValueError(
        f"Expected local_num_tokens={local_num_tokens!r} to be aligned to"
        f" t_packing={t_packing!r}."
    )
  if bt % t_packing != 0:
    raise ValueError(
        f"Expected bt={bt!r} to be aligned to t_packing={t_packing!r}."
    )
  if local_num_tokens % bt != 0:
    raise ValueError(
        f"Expected local_num_tokens={local_num_tokens!r} to be aligned to"
        f" bt={bt!r}."
    )
  if subc_quant_w1_sz is not None:
    if subc_quant_w1_sz <= 0:
      raise ValueError(
          f"Expected subc_quant_w1_sz={subc_quant_w1_sz!r} to be positive."
      )
    if subc_quant_w1_sz % 256 != 0:
      raise ValueError(
          f"Expected subc_quant_w1_sz={subc_quant_w1_sz!r} to be aligned to"
          " 256."
      )
    if hidden_size % subc_quant_w1_sz != 0:
      raise ValueError(
          f"Expected hidden_size={hidden_size!r} to be aligned to"
          f" subc_quant_w1_sz={subc_quant_w1_sz!r}."
      )
    if subc_quant_w1_sz < hidden_size:
      bd1c = subc_quant_w1_sz * t_packing
  if subc_quant_w2_sz is not None:
    if subc_quant_w2_sz <= 0:
      raise ValueError(
          f"Expected subc_quant_w2_sz={subc_quant_w2_sz!r} to be positive."
      )
    if subc_quant_w2_sz % 256 != 0:
      raise ValueError(
          f"Expected subc_quant_w2_sz={subc_quant_w2_sz!r} to be aligned to"
          " 256."
      )
    if intermediate_size % subc_quant_w2_sz != 0:
      raise ValueError(
          f"Expected intermediate_size={intermediate_size!r} to be aligned to"
          f" subc_quant_w2_sz={subc_quant_w2_sz!r}."
      )
    if subc_quant_w2_sz < intermediate_size:
      bfc = subc_quant_w2_sz
  assert (
      bf is not None
      and bd1 is not None
      and (bd2 is not None)
      and (bfc is not None)
      and (bd1c is not None)
      and (bd2c is not None)
  )
  if bfc % 128 != 0:
    raise ValueError(f"Expected bfc={bfc!r} to be aligned to 128.")
  if bd1c % (t_packing * 128) != 0:
    raise ValueError(
        f"Expected bd1c={bd1c!r} to be aligned to {t_packing * 128}."
    )
  if bd2c % (t_packing * 128) != 0:
    raise ValueError(
        f"Expected bd2c={bd2c!r} to be aligned to {t_packing * 128}."
    )
  if bf % bfc != 0:
    raise ValueError(f"Expected bf={bf!r} to be aligned to bfc={bfc!r}.")
  if bd1 % bd1c != 0:
    raise ValueError(f"Expected bd1={bd1!r} to be aligned to bd1c={bd1c!r}.")
  if bd2 % bd2c != 0:
    raise ValueError(f"Expected bd2={bd2!r} to be aligned to bd2c={bd2c!r}.")
  if hidden_size % bd1 != 0 or hidden_size % bd2 != 0:
    raise ValueError(
        f"Expected hidden_size={hidden_size!r} to be aligned to bd1={bd1!r} and"
        f" bd2={bd2!r}."
    )
  if intermediate_size % bf != 0:
    raise ValueError(
        f"Expected intermediate_size={intermediate_size!r} to be aligned to"
        f" bf={bf!r}."
    )
  if w1_scale is not None:
    assert subc_quant_w1_sz is not None
    expected_w1_scale_shape = (
        num_experts,
        2,
        hidden_size // subc_quant_w1_sz,
        1,
        intermediate_size,
    )
    if w1_scale.shape != expected_w1_scale_shape:
      raise ValueError(
          f"Expected w1_scale.shape={w1_scale.shape!r} to be"
          f" {expected_w1_scale_shape}."
      )
    if w1_scale.dtype != jnp.float32:
      w1_scale = w1_scale.astype(jnp.float32)
  if w2_scale is not None:
    assert subc_quant_w2_sz is not None
    expected_w2_scale_shape = (
        num_experts,
        intermediate_size // subc_quant_w2_sz,
        1,
        hidden_size,
    )
    if w2_scale.shape != expected_w2_scale_shape:
      raise ValueError(
          f"Expected w2_scale.shape={w2_scale.shape!r} to be"
          f" {expected_w2_scale_shape}."
      )
    if w2_scale.dtype != jnp.float32:
      w2_scale = w2_scale.astype(jnp.float32)
  if b1 is not None:
    expected_b1_shape = (num_experts, 2, 1, intermediate_size)
    if b1.shape != expected_b1_shape:
      raise ValueError(
          f"Expected b1.shape={b1.shape!r} to be {expected_b1_shape}."
      )
    if b1.dtype != jnp.float32:
      b1 = b1.astype(jnp.float32)
  if b2 is not None:
    expected_b2_shape = (num_experts, 1, hidden_size)
    if b2.shape != expected_b2_shape:
      raise ValueError(
          f"Expected b2.shape={b2.shape!r} to be {expected_b2_shape}."
      )
    if b2.dtype != jnp.float32:
      b2 = b2.astype(jnp.float32)
  if padded_num_experts != gating_output.shape[-1]:
    gating_output = jnp.pad(
        gating_output,
        ((0, 0), (0, padded_num_experts - gating_output.shape[-1])),
        constant_values=-jnp.inf,
    )
  device_kind = jax.devices()[0].device_kind.lower()
  is_tpu7x = (
      "7x" in device_kind or "tpu7x" in device_kind
  ) and "lite" not in device_kind
  tiling_cfg = FusedMoeTilingConfig.from_shapes(
      num_tokens=num_tokens,
      hidden_size=hidden_size,
      intermediate_size=intermediate_size,
      num_experts=num_experts,
      top_k=top_k,
      w_dtype=w1.dtype,
      subc_quant_w2_sz=subc_quant_w2_sz,
      is_tpu7x=is_tpu7x,
      scoring_fn=scoring_fn,
  )

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=(
          P(ep_axis_name),
          P(ep_axis_name),
          P(ep_axis_name),
          None if w1_scale is None else P(ep_axis_name),
          None if w2_scale is None else P(ep_axis_name),
          None if b1 is None else P(ep_axis_name),
          None if b2 is None else P(ep_axis_name),
          P(ep_axis_name),
      ),
      out_specs=P(ep_axis_name),
      check_vma=False,
  )
  def _sharded_jax_k_moe(
      tokens_s, w1_s, w2_s, w1_scale_s, w2_scale_s, b1_s, b2_s, gating_s
  ):
    if ep_size > 1:
      all_tokens = jax.lax.all_gather(
          tokens_s, axis_name=ep_axis_name, tiled=True, axis=0
      )
      my_id = jax.lax.axis_index(ep_axis_name)
    else:
      all_tokens = tokens_s
      my_id = jnp.int32(0)
    all_weights, all_indices = _compute_topk_routing(
        gating_s,
        num_experts=num_experts,
        top_k=top_k,
        scoring_fn=scoring_fn,
        renormalize_topk_logits=renormalize_topk_logits,
        ep_axis_name=ep_axis_name,
        ep_size=ep_size,
    )
    my_e_start = my_id * jnp.int32(local_num_experts)
    padded_tokens, slot_meta_smem, src_tok_idx, row_weight = (
        _prepare_expert_tiles(
            all_tokens,
            all_indices,
            all_weights,
            local_num_experts=local_num_experts,
            my_e_start=my_e_start,
            tile_m=tiling_cfg.tile_m,
            ep_size=ep_size,
            scoring_fn=scoring_fn,
            renormalize_topk_logits=renormalize_topk_logits,
        )
    )
    out_init = _lock_compact_layout(
        jnp.zeros(padded_tokens.shape, dtype=t_dtype)
    )
    padded_outputs = fused_moe_tile_pipeline(
        padded_tokens,
        _lock_compact_layout(w1_s),
        _lock_compact_layout(w2_s),
        slot_meta_smem,
        out_init,
        w1_scale_s,
        w2_scale_s,
        b1_s,
        b2_s,
        tile_m=tiling_cfg.tile_m,
        bf=tiling_cfg.bf,
        pallas_bf=bf,
        pallas_bd1=bd1,
        pallas_bfc=bfc,
        vmem_kib=tiling_cfg.vmem_kib,
        act_fn=act_fn,
        subc_quant_w1_sz=subc_quant_w1_sz,
        subc_quant_w2_sz=subc_quant_w2_sz,
        is_tpu7x=is_tpu7x,
    )
    padded_outputs = _lock_compact_layout(padded_outputs)
    tok_iota = jax.lax.broadcasted_iota(
        jnp.int32, (all_tokens.shape[0], padded_outputs.shape[0]), 0
    )
    if renormalize_topk_logits:
      w_mat = _lock_compact_layout(
          jnp.where(
              tok_iota == src_tok_idx[None, :],
              row_weight[None, :],
              jnp.float32(0.0),
          )
      )
      dot_prec = (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT)
    else:
      row_weight_bf16 = row_weight.astype(t_dtype)
      w_mat = _lock_compact_layout(
          jnp.where(
              tok_iota == src_tok_idx[None, :],
              row_weight_bf16[None, :],
              jnp.zeros((), dtype=t_dtype),
          )
      )
      dot_prec = None
    weighted_sum_f32 = jax.lax.dot_general(
        w_mat,
        padded_outputs,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
        precision=dot_prec,
    )
    if ep_size > 1:
      with xla_metadata.set_xla_metadata(_xla_compute_type="dense"):
        if renormalize_topk_logits:
          scattered = jax.lax.psum_scatter(
              weighted_sum_f32,
              axis_name=ep_axis_name,
              tiled=True,
              scatter_dimension=0,
          )
          return scattered.astype(t_dtype)
        return jax.lax.psum_scatter(
            weighted_sum_f32.astype(t_dtype),
            axis_name=ep_axis_name,
            tiled=True,
            scatter_dimension=0,
        )
    return weighted_sum_f32.astype(t_dtype)

  return _sharded_jax_k_moe(
      tokens, w1, w2, w1_scale, w2_scale, b1, b2, gating_output
  )


jax_k_fused_ep_moe = fused_ep_moe
fused_moe_jax_k = fused_ep_moe
