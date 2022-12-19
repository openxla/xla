# MLIR interpreter

This is a functional but partial interpreter for MLIR programs. It is intended
to support interpretation for all dialects above the LLVM dialect. The primary
intended use case is to run it as part of a compilation pipeline, to detect
miscompiles. To do this, the pass instrumentation defined in
`interpreter_instrumentation.h` can be added to a pipeline. It will attempt to
interpret the IR after each pass (on random inputs). Support for using real
inputs (e.g. from JAX) is planned.

## Current limitations

*   This has not seen much usage, so expect lots of bugs.
*   Many ops are not implemented (even basic ones like `mhlo.transpose`).
*   Many ops are only partially implemented (e.g. affine maps in memrefs)
*   The pass instrumentation can only run on random inputs, which will often not
    be sufficient to detect bugs.
