# Shapes and layout

## Structure of an XLA Op

Consider an example Op:

```
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

This consists on the following main components:

* Op Name: `fusion.3`
  * A dot or fusion op is a set of operations containing at most 1 matrix multiplication and possibly a bunch of related pointwise VPU-ops.
* Shape: `bf16[32,32,4096]`
  * This is the output shape of the op. Here the dtype is bf16 (2 bytes per parameter) and the shape is `[32,32,4096]`. More details in the following sections.
* Layout (with Tiling): `{2,1,0:T(8,128)(2,1)}`
  * This describes how the array is stored in memory. More details in the following sections.
  * `2,1,0` denotes the order of the axes in memory (column major, row major, etc.).
  * `T(8,128)(2,1)` denotes the tiling & padding used. Learn more in [Tiled Layout](tiled_layout.md).
* Memory location: `S(1)`
  * `S(1)` denotes this array lives in VMEM. More details at the end of page.
* Arguments:`bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
  * This op has one input, a bf16 array called `fusion.32` with a particular shape (as well as layout, tiling, and memory location): ` bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`. This tells us what function feeds into this one.

## Shapes

The XLA `ShapeProto` proto
([xla_data.proto](https://github.com/openxla/xla/tree/main/xla/xla_data.proto))
describes the number of dimensions, size, and data type of an N-dimensional
array (*array* in short).

### Terminology, notation, and conventions

NOTE: in the past, XLA has used the term "rank" to mean the number of dimensions
of an array. We have stopped this usage as it's inconsistent with the matrix
rank concept in linear algebra. However, you may still see the name `rank` used
in legacy documentation and some of the code.

*   The *true number of dimensions* of an array is the number of dimensions
    which have a size greater than 1.

*   Dimensions are numbered from `0` up to `N-1` for an `N` dimensional array.
    The size of a dimension is a non-negative integer. In particular, size 0 is
    valid. The dimension numbers are arbitrary labels for convenience. The
    order of these dimension numbers does not imply a particular minor/major
    ordering in the layout of the shape. The layout is determined by the
    `LayoutProto` proto.

*   By convention, dimensions are listed in increasing order of dimension
    number. For example, for a 3-dimensional array of size `[A x B x C]`,
    dimension 0 has size `A`, dimension 1 has size `B`, and dimension 2 has size
    `C`.

    Some utilities in XLA also support Python-like negative indexing: Dimension
    -1 is the last dimension (equivalent to `N-1` for an `N` dimensional array).
    For example, for the 3-dimensional array described above, dimension -1 has
    size `C`, dimension -2 has size `B`, and so on.

*   Two, three, and four dimensional arrays often have specific letters
    associated with dimensions. For example, for a 2D array:

    *   dimension 0: `y`
    *   dimension 1: `x`

    For a 3D array:

    *   dimension 0: `z`
    *   dimension 1: `y`
    *   dimension 2: `x`

    For a 4D array:

    *   dimension 0: `p`
    *   dimension 1: `z`
    *   dimension 2: `y`
    *   dimension 3: `x`

*   Functions in the XLA API which take dimensions do so in increasing order of
    dimension number. This matches the ordering used when passing dimensions as
    an `initializer_list`; e.g.

    `ShapeUtil::MakeShape(F32, {A, B, C, D})`

    will create a shape whose dimension size array consists of the sequence `[A,
    B, C, D]`.

## Layout

The `LayoutProto` proto describes how an array is represented in memory. It
includes the following fields:

```
message LayoutProto {
  repeated int64 minor_to_major;
  int64 tail_padding_alignment_in_elements;
  ...
}
```

### Minor-to-major dimension ordering

The only required field is `minor_to_major`. This field describes the
minor-to-major ordering of the dimensions within a shape. Values in
`minor_to_major` are an ordering of the dimensions of the array (`0` to `N-1`
for an `N` dimensional array) with the first value being the most-minor
dimension up to the last value which is the most-major dimension. The most-minor
dimension is the dimension which changes most rapidly when stepping through the
elements of the array laid out in linear memory.

For example, consider the following 2D array of size `[2 x 3]`:

```
a b c
d e f
```

Here dimension `0` is size 2, and dimension `1` is size 3. If the
`minor_to_major` field in the layout is `[0, 1]` then dimension `0` is the
most-minor dimension and dimension `1` is the most-major dimension. This
corresponds to the following layout in linear memory:

```
a d b e c f
```

This minor-to-major dimension order of `0` up to `N-1` is akin to *column-major*
(for 2-dimensionals). Assuming a monotonic ordering of dimensions, another way
we may refer to this layout in the code is simply "dim 0 is minor".

On the other hand, if the `minor_to_major` field in the layout is `[1, 0]` then
the layout in linear memory is:

```
a b c d e f
```

A minor-to-major dimension order of `N-1` down to `0` for an `N` dimensional
array is akin to *row-major* (for 2-dimensionals). Assuming a monotonic
ordering of dimensions, another way we may refer to this layout in the code is
simply "dim 0 is major".

#### Default minor-to-major ordering

The default layout for newly created Shapes is "dimension order is
major-to-minor" (i.e. `[N-1, ..., 0]`).

### Padding

The `tail_padding_alignment_in_elements` field defines the alignment of the
[tiled](tiled_layout.md) array in terms of the number of elements. After
applying tiling, padded elements will be added at the end of the layout until
the total number of elements is a multiple of this value.

### Indexing into arrays

The class `IndexUtil` in
[index_util.h](https://github.com/openxla/xla/tree/main/xla/index_util.h)
provides utilities for converting between multidimensional indices and linear
indices given a shape and layout. Multidimensional indices include an `int64`
index for each dimension. Linear indices are a single `int64` value which
indexes into the buffer holding the array. See `shape_util.h` and
`layout_util.h` in the same directory for utilities that simplify creation and
manipulation of shapes and layouts.

## Memory Space Identifiers

In HLO, each array may be annotated with a memory space identifier, written as S(n).

* `S(0)` (often omitted) denotes device high bandwidth memory (HBM).
* `S(1)` represents on device virtual memory (VMEM).
* `S(2)`, `S(3)`, etc., correspond to additional device specific memory spaces.
* `S(5)` indicates host memory.
