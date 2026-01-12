# Debug OOM errors with XProf

Out of memory (OOM) errors occur when your accelerator memory capacity is exhausted. This can be High Bandwidth Memory (HBM) in GPUs, and HBM, Vector Memory (VMEM), etc. in TPUs.

This page describes how to use **XProf's Memory Viewer tool** to visualize your program's memory usage, identify peak usage instances, and debug OOM errors. This involves the following steps:

1. Run your program with [`jax.profiler.trace`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html#jax.profiler.trace) to capture the profile.
2. Start XProf in the background, and use the [Memory Viewer tool](https://openxla.org/xprof/memory_viewer) to view memory utilization details.

## Example program

The following JAX program leads to an OOM error:

```python
import jax
from jax import random
import jax.numpy as jnp


@jax.profiler.trace("/tmp/xprof")
@jax.jit
def oom():
    a = random.normal(random.PRNGKey(1), (327680, 327680), dtype=jnp.bfloat16)
    return a @ a


if __name__ == "__main__":
    oom()
```

**Note:** Prefer `jax.profiler.trace` instead of `jax.profiler.start_trace`/`jax.profiler.stop_trace` because the `jax.profiler.trace` context manager handles profiling in an exception safe manner.

On a TPU-machine, this program fails with:

```shell
XlaRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=107374182400) would exceed memory (size=17179869184) :: #allocation7 [shape = 'u8[327680,327680]{1,0:T(8,128)(4,1)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of xor_convert_fusion@{}'] :: <no-hlo-instruction>
```

The error message clearly states that the output size exceed the available memory, and profile should be available at `/tmp/xprof/`.

(On a GPU-machine, the error looks like: `XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 214748364800 bytes.`)

## Run XProf

Install `xprof` (`pip install xprof`), and start an XProf instance specifying the directory where the profile is stored:

```shell
xprof --logdir=/tmp/xprof/ --port=6006
```

Go to the instance (on a local machine, at `http://localhost:6006`). In the _Tools_ dropdown, select _Memory Viewer_, and in the Memory Viewer tool window, select _HBM_ in the _Memory Types_ dropdown (usually selected by default).

<!-- TODO: Update image to one without overlapping text -->

[XProf Memory Viewer page for the above example program](images/oom_debugging_example_memory_viewer.png)

Go through [Xprof: Memory Viewer tool documentation](https://openxla.org/xprof/memory_viewer) for more details on each section.

At a high-level:
* Textual overview: Gives helpful information on peak allocation
* HLO Ops at Peak Memory Allocation Time, by Buffer Size: Shows the (left-most block) `cutlass_gemm_with_upcast` fusion operation had the highest peak memory usage.

## Further reading

Some common causes for OOM issues and debugging techniques are detailed in the [JAX documentation: GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html#common-causes-of-oom-failures). You can also refer to specific error code documentation for debugging options:
* [E1000 - Compile Time HBM OOM](./errors/error_1000.md)
* [E1001 - Compile Time Scoped Vmem OOM](./errors/error_1001.md)
