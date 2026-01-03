# Error code: 1000

**Category:** HBM OOM

This error indicates that the program requires more High Bandwidth Memory (HBM)
than is physically available on the TPU device.

**Sample Error Messages:**

```
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 49.34G of 32.00G hbm. Exceeded hbm capacity by 17.34G.

```

```
RESOURCE_EXHAUSTED: TPU TensorCore Hbm usage: 34.82G, SparseCore Hbm usage 174.10G, exceeding available bytes: 95.74G

```

**XLA Backends:** TPU

## Overview

Unlike early-abort HBM OOMs (where the compiler determines it is impossible to
fit the program on HBM early in the compilation process), these errors are
thrown near the end of the compilation. At this final stage, the
compiler performs checks to ensure that the aggregate size of all necessary
allocations fits within the device's physical memory limit.

The compiler manages the TPU's fixed HBM capacity for several types of
allocations:

* **Program Inputs and Outputs:** Training batches, optimizer states etc.
* **TensorCore + SparseCore Temporaries:** Dynamic memory required for intermediate calculations (e.g. activations, gradients etc).
* **Compiled Binary:** The machine code for both TensorCore (TC) and SparseCore (SC).
* **System Overhead:** Reserved space for the XLA Runtime (e.g. infeed buffers on older TPU generations).
* **Constants:** Constant values embedded in the HLO IR are allocated on HBM.
* **Compiler Internals:** Program level and per-HLO allocations (e.g. routing info for nodes in the mesh)

This error occurs when the XLA compiler cannot fit all of the above allocations
into the device HBM.

## Debugging

### 1. Analyze the error message and error logs

Carefully analyze the logs to determine which allocations consume the most
memory:

* **If the error explicitly lists TC and SC HBM usage:** Compare the two values.
  * **High SparseCore usage:** Investigate your embedding table sizes and sharding strategies.
  * **High TensorCore usage:** Proceed to step 2.
  * **Balanced:** If neither is individually excessive but the sum is too high, you are at the chip's capacity. You must try lowering usage of both components.

* **If the error refers to "Ran out of memory in memory space hbm":** Check the logs for a detailed breakdown of the largest allocations in HBM and proceed to step 2.

### 2. Check tensor padding and alignment

Inefficient tensor shapes are a common, silent cause of OOMs. TPUs work well
with dimensions which are multiples of 128 or 8.

* **Audit shapes of large buffers:**
  * *Example*: A shape of `(129, 1024)` might be padded to `(256, 1024)`, resulting in nearly 50% memory waste.
  * *Correction:* A shape of `(128, 1024)` requires no padding and incurs 0% memory waste.
* **Align dimensions:** Ensure all large tensor dimensions (batch size, embedding dimension, hidden size) are multiples of 128.

### 3. Adjust configuration

You can often resolve OOMs by the following relatively simple steps:

* **Reduce Batch Size:** The memory needed for intermediate activations and gradients is directly proportional to the batch size. Reducing the batch size can often help reduce memory usage.
* **Donate Input Buffers:** When using `jax.jit`, specify [donate_argnums](https://docs.jax.dev/en/latest/buffer_donation.html) for your model parameters. This allows XLA to overwrite the input memory with the output, saving significant HBM.
* **Enable Mixed Precision (bfloat16):** Use bfloat16 or quantization (int8 etc) for the largest tensors in the program if the model architecture and quality requirements allow.

### 4. Optimize architecture and sharding

If configuration changes are insufficient, the model topology might be too large
for the current hardware setup.

* **Use Newer TPU Generations:** Newer TPUs generally offer more HBM per chip, switch to newer TPU generations if available.
* **Run on a larger chip topology:** If the model weights are too large for the existing topology, you can try sharding them across more chips.
* **Implement advanced sharding techniques:**
  * Explore more advanced data, tensor, or pipeline parallelism approaches.
  * You can further provide [sharding hints](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#constraining-shardings-of-intermediates-in-jitted-code) to the compiler for intermediate values and outputs.
* **Use JAX Host Offloading:** Explore offloading opportunities for large tensors to the host CPU memory. e.g. [activation offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html#activation-offloading) and [optimizer state offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html#optimizer-state-offloading).

### 5. Tune key memory impacting XLA flags:

[Key memory flags](https://openxla.org/xla/flags_guidance#memory_flags) can be
tuned to trade-off performance for lower memory usage. But these should be used
as a last resort measure because it can adversely affect performance.

### 6. Tune XLA Rematerialization Pass / Manual Checkpointing
If the model is close to fitting into memory, you can force the XLA compiler's
rematerialization pass to prioritize memory savings, potentially at the cost of
slower compilations:
| Flag | Description | Impact / Trade-off |
| --- | --- | --- |
| `--xla_tpu_max_hbm_size_mib` | Manually sets the limit on HBM size used by the rematerialization pass. | Forces the compiler to work harder to fit the program into a limit smaller than the actual physical HBM. |
| `--xla_tpu_rematerialization_algo=PEAK_PRIORITY` | Focuses efforts at the points of peak memory usage. | Can be more efficient for aggressive memory reduction than the default algorithm. |
| `--xla_tpu_rematerialization_max_block_size_limit=32` | Controls the maximum number of instructions in a block that can be rematerialized at once. | Increasing this allows for memory savings at the cost of **significantly increases compile time**. |
| `--xla_tpu_rematerialization_block_effort_factor=10.0` | Defines the amount of effort (compile time) spent searching for blocks to rematerialize. | Higher values allow a more exhaustive search for memory savings at the cost of **increased compile times**. |
| `--xla_tpu_pre_fusion_remat=true` | Enables an additional rematerialization pass *before* the fusion pass. | Can find more memory savings, but increases compile times and may **potentially impact numerical stability**. |

You can also manually specify checkpointing hints - Use the [jax.checkpoint](https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html) decorator
with `jax.grad` to manually control which intermediates are saved on the forward
pass versus recomputed on the backward pass, trading compute cycles for HBM.

### 7. Use advanced profiling tools

[Xprof/Tensorboard Memory Viewer](https://openxla.org/xprof/memory_viewer) can help visualize the compiler's view of HBM
usage, showing peak memory allocation and the lifetime of buffers. This is
crucial for understanding what consumes HBM at the point of peak utilization.
See [xprof#getting_started](https://openxla.org/xprof#getting_started) and [tensorboard-profiling](https://docs.jax.dev/en/latest/profiling.html#xprof-tensorboard-profiling).
