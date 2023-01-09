# XLA

XLA is an open source, state-of-art compiler for machine learning that
takes models from popular frameworks such as PyTorch, TensorFlow, and JAX, and
optimizes them for high-performance execution across different hardware
platforms including GPUs, CPUs, and ML accelerators. For example, in
[BERT MLPerf submission](https://blog.tensorflow.org/2020/07/tensorflow-2-mlperf-submissions.html),
using XLA with 8 Volta V100 GPUs achieved a ~7x performance improvement
and ~5x batch size improvement (vs the same GPUs without XLA).

As a part of the OpenXLA project, XLA is built collaboratively by
industry-leading ML hardware and software companies, including
Alibaba, Amazon Web Services, AMD, Apple, Arm, Google, Intel, Meta, and NVIDIA.

## Key benefits

- [x] **Build anywhere**: XLA is already integrated into
leading ML frameworks such as TensorFlow, PyTorch, and JAX.

- [x] **Run anywhere**: It supports various backends including
GPUs, CPUs, and ML accelerators, and includes a pluggable infrastructure to add
support for more.

- [x] **Maximize and scale performance**: It optimizes a model's performance
with production-tested optimization passes and automated partitioning for model
parallelism.

- [x] **Eliminate complexity**: It leverages the power of
[MLIR](https://mlir.llvm.org/) to bring the best capabilities into a single
compiler toolchain, so you don't have to manage a range of domain-specific
compilers.

- [x] **Future ready**: As an open source project, built through a collaboration
of leading ML hardware and software vendors, XLA is
designed to operate at the cutting-edge of the ML industry.