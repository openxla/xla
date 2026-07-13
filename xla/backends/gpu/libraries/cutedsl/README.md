# CuTeDSL FFI for OpenXLA's CUDA PJRT plugin

OpenXLA's CUDA PJRT plugin registers these FFI targets:

- `__xla_gpu_cutedsl_call_v3`
- `__xla_gpu_cutedsl_call_no_cuda_graph_v3`
- `__xla_gpu_cutedsl_collective_v3`

Version 3 is the only supported contract. The two buffer-only targets accept
`module` bytes and their 32-byte SHA-256 `key`, then pass XLA buffers to the
compiled `cutlass_call` entry point using CuTeDSL's `JaxArray` layout. They
create the module during prepare and use an explicit no-op initialize stage.

The collective target accepts the same top-level `module` and `key` attributes
as the buffer-only targets, plus a `config` string containing the ProtoJSON
encoding of `CollectiveCallConfigV3`. ProtoJSON lets CuTeDSL construct the
configuration using Python's standard `json` module without a Python protobuf
dependency. Unknown JSON fields are ignored; known fields, the complete
collective configuration, and the module digest are validated during
Instantiate. The config records the clique width used to compile the
region-major peer-address table. The first FFI result is an internal
`U64[peer_region_count, abi_clique_size]` scratch buffer allocated by XLA in
device memory; remaining results are the generated function's ordinary
results. Prepare rejects a different runtime clique width before loading the
module or requesting collective resources. Initialize resolves the absolute
peer addresses, and Execute copies them into the scratch result immediately
before launching the generated function on the same stream.

The generated-function frame carries one pointer to a fixed 16-byte host
`CollectiveContextAbi` descriptor instead of one argument per peer address.
The descriptor contains the device-table pointer, rank, and clique size.
Generated host code loads only that descriptor and constructs a row-major CuTe
global-memory tensor over the table. Device kernels index the tensor to load an
absolute peer address; independent peer regions do not need to share an
allocation layout.

## Runtime linkage

XLA owns the runtime ABI used by these FFI handlers and declares its six
required C entry points in `runtime_api.h`. The function table used by the
module loader is private to XLA; the CuTeDSL runtime does not expose or
negotiate a separate API table.

Google builds select the combined static runtime through the
`cutedsl_runtime_static` label flag. The module loader calls its linked
runtime functions directly.

OSS builds compile the handlers without a link-time CuTeDSL dependency. On
first use, XLA loads `libcute_dsl_runtime.so` with `RTLD_NOW | RTLD_LOCAL`,
resolves the six required symbols, and retains the library for the process
lifetime. The platform dynamic-loader search path locates the library, so a
package or deployment can provide it through `LD_LIBRARY_PATH` or an
equivalent mechanism. Loading remains lazy and does not affect users that do
not use CuTeDSL.

Both runtime variants register CUDA helper functions directly with their ORC
JIT, so the plugin passes no runtime path in `shared_libs`. That argument
remains available for actual dependencies of generated modules.
