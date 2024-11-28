# PJRT plugin integration

## Background

This doc focuses on the recommendations about how to integrate with PJRT, and
how to test PJRT integration with JAX.

### What is PJRT

The PJRT C API provides the device compiler and runtime interface, as well as the interface between the framework and the hardware.

A PJRT plugin is a Python package that contains a shared library as a `.so` file which implements PJRT C APIs, as well as Python methods/setup to make it discoverable by the framework.

For more examples of PJRT plugins see [PJRT Examples](examples.md).

## How to integrate with PJRT

### Step 1: Implement [PJRT C API interface](https://github.com/openxla/xla/tree/main/xla/pjrt/c/pjrt_c_api.h)

**1a.** You can implement only the PJRT C API directly.

**1b (Optional)** If you're able to build against C++ code in the [xla repo](https://github.com/openxla/xla) (via forking or bazel), you can also implement the PJRT C++ API and use the C→C++ wrapper:

1. Implement a C++ PJRT client inheriting from the [base PJRT client](https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_client.h) (and related PJRT classes).
   Here are some examples of C++ PJRT client: 
   - [PJRT Example plugin](https://github.com/openxla/xla/tree/main/xla/pjrt/plugin/example_plugin)
   - [Stream executor client plugin](https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_stream_executor_client.h)
   - [CPU client plugin](https://github.com/openxla/xla/blob/main/xla/pjrt/cpu/cpu_client.h).
2. Implement a few C API methods that are not part of C++ PJRT client:
  * [PJRT\_Client\_Create](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h#L344-L365). Below is some sample pseudo code (assuming `GetPluginPjRtClient` returns a C++ PJRT client implemented above):

```
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"  // Update to the path of your local clone of xla

namespace my_plugin {
PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  std::unique_ptr<xla::PjRtClient> client = GetPluginPjRtClient();
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}
}  // namespace my_plugin
```

  Note [PJRT\_Client\_Create](https://github.com/openxla/xla/blob/b8ec2c4c9dcccaf33b548bee2c4c33a778a8cb88/xla/pjrt/plugin/example_plugin/myplugin_c_pjrt.cc#L33-L38) can take options passed from the framework. [Here is an example of how a GPU client uses this feature](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api_gpu_internal.cc#L43-L115).

  Also note that `PJRT_Client_Create` [needs extra arguments](https://github.com/openxla/xla/blob/a7d1ed8a9091bf51aed427659218548559152be2/xla/pjrt/c/pjrt_c_api_wrapper_impl.h#L459) (they can all be unimplemented methods).

  * [Optional] [PJRT\_TopologyDescription\_Create](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h#L1815-L1830).
  * [Optional] [PJRT\_Plugin\_Initialize](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h#L173-L180). This is a one-time plugin setup, which will be called by the framework before any other functions are called.
  * [Optional] [PJRT\_Plugin\_Attributes](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api.h#L182-L194).

With the [wrapper](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_wrapper_impl.h), you do not need to implement the remaining C APIs.


### Step 2: Implement GetPjrtApi

You need to implement a method `GetPjrtApi` which returns a `PJRT_Api*` containing function pointers to PJRT C API implementations. If writing in C++, this method needs to be defined as `extern C` to prevent name mangling. Below is an example assuming implementing through wrapper (similar to [pjrt\_c\_api\_cpu.cc](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_cpu.cc)):
```
const PJRT_Api* GetPjrtApi() {
  static const PJRT_Api pjrt_api =
      pjrt::CreatePjrtApi(my_plugin::PJRT_Client_Create);
  return &pjrt_api;
}
```

### Step 3: Test C API implementations

You can call [RegisterPjRtCApiTestFactory](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api_test.h#L31C6-L31C33) to run a small set of tests for basic PJRT C API behaviors.

### Step 4: Build the plugin

To build the plugin as a shared library (`.so` file) using bazel, you will also need a [BUILD file](https://bazel.build/concepts/build-files). Check out [the example plugin BUILD file](https://github.com/openxla/xla/blob/main/xla/pjrt/plugin/example_plugin/BUILD) for your reference.

Once the shared object is built, it can be found in the `bazel-out` directory. If you want to use it with JAX, you have to manually copy the shared object into the JAX plugin directory.

### Example: PJRT C API implementation

You can find an [example implementation of a PJRT plugin in the xla repo](https://github.com/openxla/xla/tree/main/xla/pjrt/plugin/example_plugin).

## How to use a PJRT plugin from JAX

### Step 1: Set up JAX

You can either use JAX nightly
```
pip install -U --pre jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
```
or [build JAX from source](https://jax.readthedocs.io/en/latest/developer.html#building-jaxlib-from-source).

You can also build a jaxlib from source at exactly the XLA commit you're building against ([instructions](https://jax.readthedocs.io/en/latest/developer.html#building-jaxlib-from-source-with-a-modified-xla-repository)).

We will start supporting ABI compatibility soon.

### Step 2: Use jax\_plugins namespace or set up entry\_point

There are two options for your plugin to be discovered by JAX.

1. Using [namespace packages](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-naming-convention). Define a globally unique module under the `jax_plugins` namespace package (i.e. just create a `jax_plugins` directory and define your module below it). Here is an example directory structure:
```
jax_plugins/
  my_plugin/
    __init__.py
    my_plugin.so
```
2. Using [package metadata](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata). If building a package via `pyproject.toml` or `setup.py`, advertise your plugin module name by including an entry-point under the `jax_plugins` group which points to your full module name. Here is an example via `pyproject.toml` or `setup.py`:
```
# use pyproject.toml
[project.entry-points.'jax_plugins']
my_plugin = 'my_plugin'

# use setup.py
entry_points={
  "jax_plugins": [
    "my_plugin = my_plugin",
  ],
}
```
Here's an example of how a PJRT plugin is implemented using Option 2: https://github.com/jax-ml/jax/tree/main/jax_plugins/cuda.

### Step 3: Implement an initialize() method

You need to implement an `initialize()` method in the `__init__.py` file of your Python module to register the plugin:
```
import os
import jax._src.xla_bridge as xb

def initialize():
  path = os.path.join(os.path.dirname(__file__), 'my_plugin.so')
  xb.register_plugin('my_plugin', priority=500, library_path=path, options=None)
```
Please refer to [the `xla_bridge.py` file](https://github.com/google/jax/blob/8f283bc9ed50d3828bd468ae57b1ee4df1527624/jax/_src/xla_bridge.py#L420) about how to use `xla_bridge.register_plugin`. It is currently a private method. A public API will be released in the future.

To verify that the plugin is registered and raise an error if it can't be loaded, you can run the following lines:
```
jax.config.update("jax_platforms", "my_plugin")
print(jax.numpy.add(1, 1))  # Any operation to trigger the plugin loading
```
JAX may have multiple backends/plugins. There are a few options to ensure your plugin is used as the default backend:
*   Option 1: run `jax.config.update("jax_platforms", "my_plugin")` in the beginning of the program.
*   Option 2: set ENV `JAX_PLATFORMS=my_plugin`.
*   Option 3: set a high enough priority when calling `xb.register_plugin` (the default value is 400 which is higher than other existing backends). Note the backend with highest priority will be used only when `JAX_PLATFORMS=''`. The default value of `JAX_PLATFORMS` is `''` but sometimes it will get overwritten.

## How to test with JAX

Some basic test cases to try:
```
# JAX 1+1
print(jax.numpy.add(1, 1))
# => 2

# jit
print(jax.jit(lambda x: x * 2)(1.))
# => 2.0

# pmap
arr = jax.numpy.arange(jax.device_count()) print(jax.pmap(lambda x: x +
jax.lax.psum(x, 'i'), axis_name='i')(arr))

# single device: [0]
# 4 devices: [6 7 8 9]
```

(We'll add instructions for running the jax unit tests against your plugin soon!)

## How to build your plugin

Below is a full walkthrough of how to build a PJRT plugin for JAX. We will build the PJRT CPU plugin from the xla repo.

```
# Install jax
~$ pip install -U "jax[cpu]"
# Build the .so file
~$ git clone https://github.com/openxla/xla
# build cpu plugin
~$ cd xla
# Build the plugin. This may take a while
~/xla$ bazel build xla/pjrt/c:pjrt_c_api_cpu_plugin.so
# Check the method exposed. It should contain `T GetPjrtApi@@VERS_1.0` at the top
~/xla$ nm -gD bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so | grep GetPjrt

# Use this plugin in JAX by setting PJRT_NAMES_AND_LIBRARY_PATHS
~/xla$ PJRT_NAMES_AND_LIBRARY_PATHS=cpu_plugin:bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so ENABLE_PJRT_COMPATIBILITY=1 TF_CPP_VMODULE=cpu_client=3,pjrt_c_api_wrapper_impl=3 TF_CPP_MIN_LOG_LEVEL=0 python
>>> import jax
>>> from jax._src import xla_bridge
# Verify that the plugin is registered
>>> jax.config.update("jax_platform_name", "cpu_plugin")
>>> client = xla_bridge.get_backend()
Platform 'cpu_plugin' is experimental and not all JAX functionality may be correctly supported!
I0000 00:00:1712356514.251375   99055 cpu_client.cc:424] TfrtCpuClient created.
2024-04-22 17:42:10.410579: I external/xla/xla/pjrt/pjrt_c_api_client.cc:134] PjRtCApiClient created.
>>> xla_bridge.backend_pjrt_c_api_version()
(0, 55)  # May vary

>>> client.platform
'cpu'
>>> client.devices()
[CpuDevice(id=0), CpuDevice(id=1), CpuDevice(id=2), CpuDevice(id=3)]

# Use the plugin from JAX
>>> import numpy as np
>>> x = np.arange(12.).reshape((3, 4)).astype("float32")
>>> device_x = jax.device_put(x)
>>> @jax.jit
... def fn(x):
...     return x * x
>>> x_shape = jax.ShapeDtypeStruct(shape=(16, 16), dtype=jax.numpy.dtype('float32'))
>>> lowered = fn.lower(x_shape)
>>> executable = lowered.compile()._executable  # Will output info about the HloModule
>>> executable.as_text()
'HloModule jit_fn, entry_computation_layout={(f32[16,16]{1,0})->f32[16,16]{1,0}}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}\n\nENTRY %main.3 (Arg_0.1: f32[16,16]) -> f32[16,16] {\n  %Arg_0.1 = f32[16,16]{1,0} parameter(0)\n  ROOT %multiply.2 = f32[16,16]{1,0} multiply(f32[16,16]{1,0} %Arg_0.1, f32[16,16]{1,0} %Arg_0.1), metadata={op_name="jit(fn)/jit(main)/mul" source_file="<stdin>" source_line=3}\n}\n\n'

# JAX 1+1
>>> jax.numpy.add(1, 1)
Array(2, dtype=int32, weak_type=True)

# jit
>>> jax.jit(lambda x: x * 2)(1.)
Array(2., dtype=float32, weak_type=True)

# pmap (4 devices in this example)
>>> arr = jax.numpy.arange(jax.device_count())
>>> jax.pmap(lambda x: x + jax.lax.psum(x, 'i'), axis_name='i')(arr)
Array([6, 7, 8, 9], dtype=int32)
```
