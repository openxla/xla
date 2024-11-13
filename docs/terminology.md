# XLA Terminology

There are several terms that are used in the context of XLA, MLIR, LLVM, and other related technologies. Below is a partial list of these terms and their definitions.

- **OpenXLA**
    - OpenXLA is an open ecosystem of performant, portable, and extensible machine learning (ML) infrastructure
      components that simplify ML development by defragmenting the tools between frontend frameworks and hardware
      backends. It includes the XLA compiler, StableHLO, VHLO, [PJRT](https://openxla.org/xla/pjrt/overview) and other
      components.
- **XLA**
    - XLA (Accelerated Linear Algebra) is an open source compiler for machine learning. The XLA compiler takes models
      from popular frameworks such as PyTorch, TensorFlow, and JAX, and optimizes the models for high-performance
      execution across different hardware platforms including GPUs, CPUs, and ML accelerators. The XLA compiler outputs
      some code to LLVM, some to "standard" MLIR, and some to [Triton MLIR](https://triton-lang.org/main/dialects/dialects.html)
      that is processed by (MLIR-based) OpenAI Triton compiler.
- **PJRT**
    - [PJRT](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) is a uniform Device API that simplifies
      the growing complexity of ML workload execution across hardware and frameworks. It provides a hardware and
      framework independent interface for compilers and runtimes.
- **StableHLO**
    - StableHLO the public interface to OpenXLA, it is a standardized MLIR dialect that may be used by different
      frameworks and compilers in the OpenXLA ecosystem. XLA supports StableHLO, and immediately converts it to HLO on
      the input. There are some [StableHLO to StableHLO](https://openxla.org/stablehlo/generated/stablehlo_passes)
      passes that use MLIR framework work. It is also possible to convert StableHLO to other compilers' IR without using
      HLO, for example in cases where an existing IR is more appropriate.
- **CHLO**
    - CHLO is a collection of higher level operations which are optionally decomposable to StableHLO.
- **VHLO**
    - The [VHLO Dialect](https://openxla.org/stablehlo/vhlo) is a MLIR dialect that is a compatibility layer on top of
      StableHLO. It provides a snapshot of the StableHLO dialect at a given point in time by versioning individual
      program elements, and is used for serialization and stability.
- **MHLO**
    - MHLO aka [MLIR-HLO](https://github.com/tensorflow/mlir-hlo) is a standalone MLIR-based input of XLA, but is
      deprecated. Users are encouraged to use StableHLO instead.
- **HLO**
    - HLO is an internal graph representation (IR) for the XLA compiler (and also supported input). It is **not** based
      on MLIR, and has it's own textual syntax and binary (protobuf based) representation.
- **MLIR**
    - [MLIR](https://mlir.llvm.org) is intended to be a hybrid IR which can support multiple different requirements in a
      unified infrastructure. It's accepted by LLVM as an input but it is not a language in itself; it is a framework
      that allows you to define your own dialect, and then compile it to LLVM code.
- **LLVM**
    - [LLVM](https://llvm.org/) is a compiler backend, and a language that it takes as an input. Many compilers, like
      clang or rust, generate LLVM code as a first step, and then LLVM generates machine code from it. This allows
      developers to reuse code that is similar in different compilers, and also makes supporting different target
      platforms easier.
- **Dialects**
    - [Dialects](https://mlir.llvm.org/docs/LangRef/#dialects) are the mechanism by which to engage with and extend the
      MLIR ecosystem. There are [several standard dialects provided by LLVM](https://mlir.llvm.org/docs/Dialects/), most
      important of which is [LLVM IR](https://mlir.llvm.org/docs/Dialects/LLVM/), which has one-to-one mapping to LLVM
      code. The usual flow of a compiler is to start with MLIR in its own dialect, then iteratively lower to standard
      LLVM dialects, until you reach "LLVM IR", which is converted to LLVM code and processed further by LLVM compiler.
