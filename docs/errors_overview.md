# XLA Errors Overview

XLA errors are categorized into different XLA error sources. Each source has
a list of an additional context other than the error message, which will be
attached to each error within the category.

### E01xx - Runtime data transfer/allocation failures

Failures when allocating memory on accelerator (such as HBM or Vmem) or host’s memory (DRAM):
- Memory space,
- Available memory,
- Summarized view of what other large buffers/programs are occupying memory,
- Backtrace - stacktrace at the point where the error is thrown.

### E03xx - Runtime Program Execution Failure due to Hardware Detected Program/User Errors

Accelerator hardware - both TPUs (V*C program errors) and GPUs (CUDA runtime errors)
can detect illegal instructions in the generated program and move to a halt state.

### E20xx - Compile Time Mosaic Deserialization Failure

These are errors which happen during parsing of Mosaic kernels:
- Compiler device type (TPU - TC or SC, GPU, CPU, etc.),
- User Python source line number,
- Framework Named Scope,
- `HloModule` name,
- Most relevant `HloInstruction` name (the Mosaic CustomCall),
- C++ source line number.

### E21xx - Compile Time Mosaic Internal Error

These are Mosaic errors due to internal precondition/assumption check failures:
- Compiler device type (TPU - TC or SC, GPU, CPU, etc.),
- User Python source line number,
- Framework Named Scope,
- `HloModule` name,
- Most relevant `HloInstruction` name (the Mosaic CustomCall),
- C++ source line number.

### E22xx - Compile Time Mosaic User Error

These are errors that happen during Mosaic kernel compilation as a result of
an invalid program:
- Compiler device type (TPU - TC or SC, GPU, CPU, etc.),
- User Python source line number,
- Framework Named Scope,
- `HloModule` name,
- Most relevant `HloInstruction` name (the Mosaic CustomCall),
- C++ source line number.

## Error codes

Here is an index list with all [error codes](error_codes.md).
