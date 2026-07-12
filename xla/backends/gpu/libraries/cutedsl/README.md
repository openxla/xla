# CuTeDSL FFI for OpenXLA's CUDA PJRT plugin

OpenXLA's CUDA PJRT plugin registers these FFI targets:

- `__xla_gpu_cutedsl_call_v3`
- `__xla_gpu_cutedsl_call_no_cuda_graph_v3`
- `__xla_gpu_cutedsl_collective_v3`

Version 3 is the only supported contract. The two buffer-only targets accept
`module` bytes and their 32-byte SHA-256 `key`, then pass XLA buffers to the
compiled `cutlass_call` entry point using CuTeDSL's `JaxArray` layout. They
create the module during prepare and use an explicit no-op initialize stage.

The collective target accepts one `config` string containing the ProtoJSON
encoding of `CollectiveCallConfigV3`. ProtoJSON lets CuTeDSL construct the
configuration using Python's standard `json` and `base64` modules without a
Python protobuf dependency. Unknown JSON fields are ignored; known fields and
the complete collective configuration are validated during Instantiate. The
config records the clique width used to compile the region-major peer-address
table and includes the base64-encoded module image. The first FFI result is an
internal
`U64[peer_region_count, abi_clique_size]` scratch buffer allocated by XLA in
device memory; remaining results are the generated function's ordinary
results. Prepare rejects a different runtime clique width before loading the
module or requesting collective resources. Initialize resolves the absolute
peer addresses, and Execute copies them into the scratch result immediately
before launching the generated function on the same stream.

The generated-function frame carries one pointer to a fixed 16-byte host
`CollectiveContextAbiV3` descriptor instead of one argument per peer address.
The descriptor contains the device-table pointer, rank, and clique size.
Generated host code loads only that descriptor and constructs a row-major CuTe
global-memory tensor over the table. Device kernels index the tensor to load an
absolute peer address; independent peer regions do not need to share an
allocation layout.

## Runtime linkage

The shared module loader directly includes `CuteDSLRuntime.h` and depends on
the `//xla/backends/gpu/libraries/cutedsl:cutedsl_runtime` label flag; both FFIs
use that target transitively. Builds must select a C++ provider that exports
the canonical header and either the shared or combined-static runtime from the
same CuTeDSL revision:

```text
--//xla/backends/gpu/libraries/cutedsl:cutedsl_runtime=@cutedsl_runtime//:runtime
```

Without a configured provider, the runtime-dependent targets are incompatible
and are not registered in the CUDA PJRT plugin.

A shared provider can wrap `libcute_dsl_runtime.so` in a `cc_import`.
Shared-runtime wheel builds must also identify the provider's one-file DSO
target:

```text
--//xla/backends/gpu/libraries/cutedsl:cutedsl_runtime=@cutedsl_runtime//:runtime
--//xla/backends/gpu/libraries/cutedsl:cutedsl_runtime_is_shared=true
--//xla/backends/gpu/libraries/cutedsl:cutedsl_runtime_dso=@cutedsl_runtime//:runtime_dso
```

The `cutedsl_runtime_dso` target must provide exactly one file. The build emits
it as `libcute_dsl_runtime.so` beside the plugin in both Bazel runfiles and the
CUDA PJRT wheel, so the plugin's `$ORIGIN` RUNPATH resolves the runtime SONAME.
The link provider and DSO must come from the same CuTeDSL build. Analysis fails
if the shared/static mode and DSO selection disagree.

A combined-static provider needs only `cutedsl_runtime`; leave
`cutedsl_runtime_dso` unset so the wheel contains no additional runtime
artifact. Copying a dynamically linked plugin without its runtime is not a
complete deployment.

The linked runtime registers CUDA helper functions directly with its ORC JIT,
so the plugin passes no runtime path in `shared_libs`. That argument remains
available for actual dependencies of generated modules.
