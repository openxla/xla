# Error code: 1000

**Category:** High Bandwidth Memory (HBM) Out of Memory (OOM)

**Type:** Compile time

**XLA Backends:** TPU

## Error log examples

```
RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 49.34G of 32.00G hbm. Exceeded hbm capacity by 17.34G.
```
```
RESOURCE_EXHAUSTED: Compilation failure: Aborting compilation early because it's unlikely to have enough memory. Requires 3.84T, has 90.96G available. If more detailed logging is desired, set --xla_tpu_impure_oom_fast_exit_threshold=-1
```

## Why do these happen?

The XLA compiler throws these errors when the total High Bandwidth Memory (HBM) required at any specific point in the compiled program exceeds the physical capacity of the TPU device.

To effectively debug these errors, it is crucial to understand that XLA allocates HBM for more than just your model weights. The available HBM is shared by a number of different functions, including:
* **Parameters & Optimizer State:** The static weights, embeddings, and optimizer slot variables (e.g., momentum, variance)
* **Intermediate Values:** Dynamic memory required for activation tensors, gradients, and temporary buffers during the forward/backward pass
* **Program Code:** The actual executable binary of the compiled XLA program
* **System Overhead:** A small portion of memory is reserved for XLA Runtime internals

This error is thrown if the XLA compiler is unable to fit all of the above allocations on the device HBM.

## How can a user fix their program when they do happen?

1. **Reduce Batch Size:** The memory needed for intermediate activations and gradients is directly proportional to the batch size. Reducing the batch size can often help reduce memory usage.
2. **Donate Input Buffers:** When using `jax.jit`, [specify `donate_argnums`](https://docs.jax.dev/en/latest/buffer_donation.html) for your model parameters. This allows XLA to overwrite the input memory with the output, saving significant HBM.
3. **Enable Mixed Precision (bfloat16)**: Use bfloat16 or quantization (int8, etc.) for the largest tensors in the program if the model architecture and quality requirements allow.
4. **Use Newer TPU Generations:** Newer TPUs generally offer more HBM per chip, switch to newer TPU generations if available.
5. **Run on a larger chip topology:** If the model weights are too large for the existing topology, you can try sharding them across more chips.
6. **Explore better sharding techniques:**
    1. Explore more advanced data, tensor, or pipeline parallelism based on the program’s peak memory usage.
    2. You can further provide [sharding hints](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#constraining-shardings-of-intermediates-in-jitted-code) to the compiler for intermediate values and outputs.
7. **Memory vs. Compute Trade-offs:**
    1. **Use JAX Host Offloading: Explore offloading opportunities for large tensors to the host CPU memory. e.g. [activation offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html#activation-offloading) and [optimizer state offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html#optimizer-state-offloading).
    2. **Manual remat specification:** Use the [`jax.checkpoint` decorator](https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html) with `jax.grad` to manually control which intermediates are saved on the forward pass versus recomputed on the backward pass, trading off memory and FLOPs.
    3. Tweaking XLA Rematerialization Strategy: If your model is on the edge of fitting into memory, you can force the XLA compiler to be more aggressive with rematerialization. By default, XLA balances compilation time and runtime performance against memory usage. But the following flags shift that balance to prioritize memory savings, potentially at the cost of slower execution:
        * `--xla_tpu_max_hbm_size_mib`: Manually set the limit on HBM size used by the rematerialization pass to force it to work harder to fit the program into a limit much smaller than the actual HBM limit.
        * `--xla_tpu_rematerialization_algo=PEAK_PRIORITY`: Focuses rematerialization efforts at the points of peak memory usage. This can be more efficient for aggressive memory reduction.
        * `--xla_tpu_rematerialization_max_block_size_limit=32`: Controls the maximum number of instructions in a block that the algorithm will consider rematerializing at once. Increasing this value allows the algorithm to explore more complex rematerialization opportunities, potentially leading to better memory savings, but significantly increasing compile time.
        * `--xla_tpu_rematerialization_block_effort_factor=10.0`:  Determines the amount of effort (compile time) the algorithm will spend searching for blocks to rematerialize. Higher values allow a more exhaustive search at the cost of increased compile times.
        * `--xla_tpu_pre_fusion_remat=true`: Enables an additional rematerialization pass before the fusion pass. This can sometimes find more opportunities to save memory, but at the cost of increased compile times and potentially impact to numerical stability.
    4. **Explore memory-efficient model architectures:** If possible, explore alternative model architectures or layer types that are inherently more memory-efficient.


## How can a user debug these failures?

* **[Xprof/Tensorboard Memory Viewer](https://openxla.org/xprof/memory_viewer):** Visualizes the compiler's view of HBM usage, showing peak memory allocation and the lifetime of buffers. This is crucial for understanding what consumes HBM at the point of peak utilization. See [XProf documentation on Getting Started](https://openxla.org/xprof#getting_started) and [JAX documentation on XProf (TensorBoard profiling)](https://docs.jax.dev/en/latest/profiling.html#xprof-tensorboard-profiling).
* **On "Aborting compilation early ..." error:** This occurs when the XLA compiler aborts early in the compilation process if it determines that its mathematically impossible to fit the model, regardless of optimization strategies. If you need to see exactly which tensors or operations are pushing memory over the limit, you can force the compiler to not abort early by setting the flag: `--xla_tpu_impure_oom_fast_exit_threshold=-1`. This will likely result in a standard OOM later in the compilation, but it will generate a detailed failure log describing the memory state.
