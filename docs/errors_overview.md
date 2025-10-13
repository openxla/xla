# XLA Errors Overview

This

## Categories

XLA errors are categorized into different XLA error sources. Each source has
a list of an additional context other than the error message, which will be
attached to each error within the category.

### 0xxx - Compile Time OOM

These errors are a result of the compiled program needing more memory than
is available.

- Compiler device type (TPU - TC or SC, GPU, CPU, etc.)
- `HloModule` name
- Memory space

### 1xxx - Compile Time Internal Error

These errors usually come from `CHECK` failures from failed
preconditions/assumptions.

- Compiler device type (TPU - TC or SC, GPU, CPU, etc.)
- User Python source line number
- Framework Named Scope
- `HloModule` name
- Most relevant `HloInstruction` name
- C++ source line number
- `HloPassPipeline` name (if in HLO pass)
- `HloPass` name (if in HLO pass)
- LLO, MLIR, & LLVM passses as well

### 2xxx - Compile Time Unimplemented

Errors arising from the fact that given operation is not yet implemented
for a given device.

- Compiler device type (TPU - TC or SC, GPU, CPU, etc.)
- User Python source line number
- Framework Named Scope
- `HloModule` name
- Most relevant `HloInstruction` name
- C++ source line number
- `HloPassPipeline` name (if in HLO pass) or LLO, MLO, LLVM, Shardy
  propagation passes
- `HloPass` name (if in HLO pass)

## Error codes

Here is an index list with all [error codes](error_codes.md).
