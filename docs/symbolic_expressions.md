# Symbolic expressions and maps

Symbolic expressions (`SymbolicExpr`) and maps (`SymbolicMap`) are mathematical abstraction systems that enable symbolic tensor computations and transformations in the compilation pipeline.
They act as the "mathematical bridge" between a high-level HLO operation and the actual memory addresses accessed by the GPU/CPU.

`SymbolicExpr` and `SymbolicMap` XLA's custom implementations that supersede the legacy `mlir::AffineExpr` and `mlir::AffineMap`.

## `SymbolicExpr`

`SymbolicExpr` represents mathematical expressions as an abstract syntax tree, with *dimensions* and *symbols* that are not resolved until later in the compilation pipeline.

Example of a `SymbolicExpr`: `d0 + s0 * 8`

It allows XLA to perform symbolic algebra on tensor shapes, and enables calculating how [tiled data](./tiled_layout.md) changes after specific operations.

Supported operations: `add`, `multiply`, `mod`, `floordiv`, `ceildiv`, `min`, `max`.

*Note: `min`, and `max` do not have `AffineExpr` counterparts.*

## `SymbolicMap`

`SymbolicMap` is a collection and combination of `SymbolicExprs`. It's a mathematical mapping of transformation between coordinate systems, typically between input and output tensors.

Example of a `SymbolicMap`: `(d0, d1)[s0, s1] -> ((d0 + s0), (d1 * s1))`

`SymbolicMap` forms the mathematical basis for `IndexingMap`, that describes how tensor elements map to each other in [HLO semantics](./operation_semantics.md).

`IndexingMap` consists on symbolic maps with domain-specific constraints. It enables shape and tiling analysis, and collapsing operation chains (like consecutive reshaping, transpose, broadcast, etc.) into optimized indexing calculation.

Learn more with concrete examples in [Indexing Analysis](./indexing.md).
