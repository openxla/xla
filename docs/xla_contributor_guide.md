# XLA Contributor Onboarding Guide

This guide is a practical map of the OpenXLA XLA repository for new
contributors. It explains what XLA does, how its main pieces fit together, how
to build and inspect a small computation, and how to start a code contribution.
It is an orientation guide, not a replacement for the detailed documentation
linked throughout.

> **AI assistant context:** When using an AI coding assistant, give it this
> guide as orientation, but ask it to inspect the current checkout before
> making claims or edits. The root [`AGENTS.md`](../AGENTS.md),
> [`xla/AGENTS.md`](../xla/AGENTS.md),
> the relevant component's existing code and tests, and the canonical docs take
> precedence. You are responsible for understanding and validating any generated
> change; see [AI-generated contributions](contributing.md#ai-generated-contributions).

## 1. What this repository is

XLA (Accelerated Linear Algebra) is a compiler and runtime infrastructure for
machine-learning computations. It accepts computations from frontends such as
JAX, TensorFlow, and PyTorch/XLA, optimizes them, and produces programs for
supported devices and backends.

This repository is primarily for people developing or integrating the compiler.
It is not a graphical application, a model-training framework, or usually the
right thing to build just to use XLA from a model. To use XLA through a
framework, start with that framework's documentation. To change compiler
behavior, develop a backend, or debug compilation, this repository is the right
place.

The compiler's broad goals are to improve execution speed and memory use,
reduce framework/runtime overhead, and let computations target different
hardware. See the [architecture overview](architecture.md) for additional
context.

## 2. A mental model of compilation

The exact path depends on the frontend and backend, but the following model is
useful:

1. **Frontend captures a computation.** A framework traces or otherwise
   represents the operations and tensor shapes of a program.
2. **Portable and compiler IRs represent the computation.** StableHLO is a
   versioned, portable operation set used at framework/compiler boundaries.
   XLA also uses its internal HLO representation for many compiler analyses
   and transformations. These are compiler IRs, not ordinary source code.
3. **Optimization and analysis transform the graph.** Passes simplify,
   canonicalize, fuse, legalize, schedule, and analyze the computation.
   Some passes apply across backends; others are target-specific. Analysis
   includes decisions such as layouts, memory use, and partitioning.
4. **The backend lowers the computation.** The selected backend applies
   hardware-specific transformations and generates code or commands suitable
   for that target. CPU and GPU code generation commonly involve LLVM; GPU
   compilation can also involve GPU-specific code generation and libraries.
5. **A runtime loads and executes the result.** Buffers and executable programs
   are managed by a runtime interface. PJRT is XLA's uniform device/runtime
   API and is used by framework integrations and plugins.

An important boundary: a frontend describes a computation, the compiler
transforms and lowers it, and the runtime manages device execution. Not every
program follows one single entry point or identical sequence of passes.

### `Run` and `RunImpl` are not the whole compiler's execution API

Names such as `Run` and `RunImpl` are local to the class that defines them.
For example, in
[`HloPassInterface`](../xla/hlo/pass/hlo_pass_interface.h), `Run()` is the
public operation for running one HLO pass on a module. It dispatches to the
protected virtual `RunImpl()` implemented by a concrete pass. That is a pass
framework convention; it is not the universal entry point for compiling or
executing a user's model.

At the PJRT boundary, [`PjRtClient::Compile`](../xla/pjrt/pjrt_client.h)
and `CompileAndLoad` represent compilation/loading operations, while
[`PjRtLoadedExecutable::Execute`](../xla/pjrt/pjrt_client.h) executes an
already loaded executable. When following a `Run` or `RunImpl` symbol, first
identify its containing class and follow that class's callers.

## 3. Where to look in the source tree

The repository is large. Start from the component that owns the behavior rather
than reading the tree from top to bottom.

| Area | What to look for |
| --- | --- |
| `xla/hlo/` | HLO representation, pass framework, analyses, and shared HLO transformations |
| `xla/service/` | Compiler services, pipelines, and many compiler/backend components |
| `xla/backends/` | Newer backend-specific implementations; check the relevant subdirectory |
| `xla/pjrt/` | PJRT client, executable, buffer, plugin, and runtime interfaces |
| `xla/stream_executor/` | Device and execution abstractions used by backends |
| `xla/tools/` | Command-line utilities such as HLO runners and compiler inspection tools |
| `xla/tsl/` | Platform and support code shared with the TensorFlow/XLA ecosystem |
| `docs/` | Architecture, build, debugging, tooling, and contribution guides |
| `third_party/` | Dependency declarations, patches, and repository integration |

These are orientation hints, not strict ownership boundaries; the codebase is
evolving and related code can live in multiple directories. Use `rg` or your
editor's symbol search to find the implementation, tests, and BUILD target
together. Many tests live near the implementation.

## 4. Set up and build

### Recommended starting point

Follow the [developer guide](developer_guide.md) and
[build-from-source instructions](build_from_source.md) for the full, current
prerequisites. Bazelisk is recommended because it selects the Bazel version
expected by the repository. The first build downloads and compiles a large
dependency graph (including compiler infrastructure such as LLVM and MLIR), so
it may take a while and require substantial disk space.

### Windows in this checkout

Native Windows builds are CPU-only and require a Bash environment such as Git
Bash or MSYS2, Python, Clang/`clang-cl`, Visual Studio C++ tools, and permission
for Bazel to create symlinks. Enable Windows Developer Mode or use an
appropriately elevated shell if Bazel reports a symlink permission error. The
repository's documented native Windows build command is:

```bash
cd /d/xla
bazel --version

bazel build \
  --config=xla_windows_x86_cpu_2022 \
  --keep_going \
  --build_tag_filters=-no_oss,-oss_excluded,-gpu,-no_windows,-windows_excluded \
  -- //xla/... -//xla/hlo/experimental/... -//xla/python_api/... -//xla/python/...
```

Run this in Git Bash/MSYS2, not PowerShell. CUDA is not supported by the native
Windows build; use WSL2 or Linux for GPU work. If the Windows build does not
support a particular tool or test target, try the same workflow in WSL2/Linux.

### Linux CPU build with Docker

From the repository root in a Linux shell with Docker installed, the
documented `ml-build` container workflow is:

```bash
docker run -itd --rm \
  --name xla \
  -w /xla \
  -v "$PWD:/xla" \
  us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest \
  bash

docker exec xla ./configure.py --backend=CPU
docker exec xla bazel build \
  --spawn_strategy=sandboxed \
  --test_output=all \
  //xla/...
```

The container shares the checkout at `/xla`, so source edits made on the host
are visible in the container. GPU configuration and non-Docker setup are
described in [Build from source](build_from_source.md).

## 5. Run and inspect a tiny computation

XLA does not open a visual application window. A useful way to see the compiler
work is to feed a small HLO module to the repository's command-line runner.
After a compatible CPU toolchain is configured, create `sample.hlo` in the
repository root:

```text
HloModule add

ENTRY main {
  lhs = f32[2] parameter(0)
  rhs = f32[2] parameter(1)
  ROOT result = f32[2] add(lhs, rhs)
}
```

Run it from the repository root:

```bash
bazel run //xla/tools:run_hlo_module -- \
  --platform=CPU \
  --reference_platform=Interpreter \
  sample.hlo
```

`bazel run` builds the tool if needed and launches it. The runner compiles and
executes the HLO and, by default with the reference option above, compares the
result with the interpreter. On platforms where this tool target is not
available, use the tool workflow in WSL2/Linux. See [XLA tooling](tools.md)
for flags and additional tools.

To inspect a compilation stage rather than run the module:

```bash
bazel run //xla/tools:hlo-opt -- \
  --platform=cpu \
  --stage=hlo \
  sample.hlo
```

`hlo-opt` has platform- and stage-dependent outputs; use
`--list-stages` to see the choices supported by the selected platform. For a
real framework program, set its `XLA_FLAGS` environment variable to include an
HLO dump directory, for example `--xla_dump_to=/tmp/xla-dump`, then inspect the
emitted HLO and related compiler artifacts. The exact environment-variable
syntax varies by shell and framework; see [HLO dumps](hlo_dumps.md).

## 6. Make a focused first contribution

1. **Pick a narrow issue.** A documentation fix, a well-scoped HLO pass change,
   or a small bug with a reproducible example is easier to understand and
   review than a broad refactor.
2. **Find the owning code and target.** Search for the symbol or behavior, read
   nearby implementation and tests, and inspect its Bazel `BUILD` target.
3. **Reproduce the behavior.** Reduce the problem to a small HLO or focused
   unit test where possible. For a bug fix, the project's guidance is to first
   add a test that fails for the expected reason.
4. **Make a small change.** Follow local patterns and the
   [XLA-specific code guidelines](../xla/AGENTS.md). Avoid unrelated cleanup.
5. **Run targeted validation.** Build or test the smallest relevant target,
   then expand validation if the change affects more components.
6. **Review and explain the change.** Check the diff, explain why the change
   is needed, describe the design and trade-offs, and state what you tested.

For example, one hardware-independent unit test target documented in the XLA
guidelines can be run with:

```bash
bazel test //xla/service:hlo_verifier_test
```

Use the actual test target next to the code you changed; the example is not a
substitute for identifying the relevant test. HLO pass tests often use
FileCheck-style expected output. See [Testing HLO passes](test_hlo_passes.md).
GPU and other hardware-dependent tests may require a configured backend and
working devices.

Before opening a pull request, read [Contributing](contributing.md). XLA asks
for focused changes, appropriate tests and benchmark evidence for performance
changes, and clear descriptions. AI-assisted contributions must be understood
by their author and should follow the project's specific AI contribution
guidance. Follow the repository's pull request template if you open a PR.

## 7. A suggested learning path

1. Read this guide and the [architecture overview](architecture.md).
2. Build or run one tool from [XLA tooling](tools.md), starting with the small
   HLO example above.
3. Read [HLO passes](hlo_passes.md) if you are interested in compiler
   transformations, or [Developing a new backend](developing_new_backend.md)
   if you are integrating new hardware.
4. Trace one real operation through its HLO, relevant pass, backend lowering,
   and tests rather than trying to understand the entire repository at once.
5. Choose a small issue, write or update a targeted test, and follow the
   [contribution process](contributing.md).

## Canonical references

- [XLA architecture](architecture.md)
- [Developer guide](developer_guide.md)
- [Build from source](build_from_source.md)
- [XLA tooling](tools.md)
- [HLO passes](hlo_passes.md)
- [Testing HLO passes](test_hlo_passes.md)
- [Developing a new backend](developing_new_backend.md)
- [Contributing](contributing.md)
- [`xla/AGENTS.md`](../xla/AGENTS.md)
