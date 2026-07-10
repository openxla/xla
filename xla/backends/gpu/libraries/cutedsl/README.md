# CuTeDSL XLA FFI

OpenXLA registers these CUDA FFI targets:

- `__xla_gpu_cutedsl_call_v3`
- `__xla_gpu_cutedsl_call_no_cuda_graph_v3`

Version 3 is the only supported contract. It accepts `module` bytes and their
32-byte SHA-256 `key`, creates the module during prepare, runs an explicit no-op
initialize stage, and passes XLA buffers to the compiled `cutlass_call` entry
point using CuTeDSL's `JaxArray` layout.

## Runtime linkage

The `//xla/backends/gpu/libraries/cutedsl:cutedsl_runtime` label flag selects a
Bazel C++ provider for the six `CuteDSLRT_*` functions. An OSS build defaults
to an unavailable provider so normal XLA binaries remain linkable. Select
exactly one provider when building a plugin that will execute the v3 FFI:

```text
--//xla/backends/gpu/libraries/cutedsl:cutedsl_runtime=@cutedsl_runtime//:runtime_shared
```

The provider can wrap either:

- `libcute_dsl_runtime.so` in a `cc_import`; or
- DKG's standalone combined `libcute_dsl_runtime.a` in an always-linked
  `cc_import`, together with cudart, `dl`, `m`, and pthread link dependencies.

Do not use `libcute_dsl_runtime_static.a` by itself. It does not include the
runtime's private LLVM, host-runtime, and CUDA-dialect dependency closure. The
public Python wheel currently supplies the shared library but not the combined
archive.

Both standalone runtime variants register the CUDA helper functions directly
with their ORC JIT. OpenXLA therefore passes no runtime path in `shared_libs`;
that argument remains available for actual dependencies of generated modules.

A shared provider makes `libcute_dsl_runtime.so` a load-time dependency of the
PJRT plugin. Bazel supplies a runfiles-relative RUNPATH for Bazel-built tests.
An installed plugin must also package the DSO and establish a stable RUNPATH,
or document an `LD_LIBRARY_PATH` requirement. Copying the plugin alone is not a
complete dynamic deployment.
