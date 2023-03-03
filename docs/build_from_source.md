# Build from source

This document describes how to build XLA components.

If you did not clone the XLA repository or install Bazel, please check out the "Get started" section of the README document.

## Linux


### Configure

XLA builds are configured by the `.bazelrc` file in the repository's
root directory. The `./configure` or `./configure.py` scripts can be used to
adjust common settings.

If you need to change the configuration, run the `./configure` script from
the repository's root directory. This script will prompt you for the location of
XLA dependencies and asks for additional build configuration options
(compiler flags, for example). Refer to the _Sample session_ section for
details.

```
./configure
```

There is also a python version of this script, `./configure.py`. If using a
virtual environment, `python configure.py` prioritizes paths
within the environment, whereas `./configure` prioritizes paths outside
the environment. In both cases you can change the default.

### CPU support

If you are using [TensorFlow's docker container](https://www.tensorflow.org/install/docker) you can build XLA with CPU support using the following command:

```
docker exec xla ./configure && bazel build //xla/...  --spawn_strategy=sandboxed --nocheck_visibility --test_output=all
```

### GPU support

We recommend using a GPU docker container to build XLA with GPU support, such as:

```
docker run --name xla_gpu -w /xla -it -d --rm -v $PWD:/xla tensorflow/tensorflow:devel-gpu bash
```

Now you can build XLA with GPU support using the following command:

```
docker exec -e TF_NEED_CUDA=1 xla_gpu ./configure && bazel build --test_output=all --spawn_strategy=sandboxed --nocheck_visibility //xla/...
```

For more details regarding [TensorFlow's GPU docker images you can check out this document.](https://www.tensorflow.org/install/source#gpu_support_3)