# Contribute to XLA

This guide shows you how to get started developing the XLA project. First go to
[CONTRIBUTING.md](../CONTRIBUTING.md), review the contribution process, and, if
you haven't already done so, sign the
[Contributor License Agreement](https://cla.developers.google.com/about). Then
follow the steps below to get and build the source code.

This guide assumes that you're familiar with
[Git](https://github.com/openxla/xla) and [GitHub](https://docs.github.com/).

## Get the code

1. Create a fork of the [XLA repository](https://github.com/openxla/xla).
2. Clone your fork of the repo, replacing `<USER>` with your GitHub username:
   ```sh
   git clone https://github.com/<USER>/xla.git
   ```
3. Change into the `xla` directory:
   ```sh
   cd xla
   ```
4. Configure the remote upstream repo:
   ```sh
   git remote add upstream https://github.com/openxla/xla.git
   ```

## Set up an environment

To build XLA, you must have [Bazel](https://bazel.build/install) installed. The
recommended way to install Bazel is using
[Bazelisk](https://github.com/bazelbuild/bazelisk#readme), which automatically
downloads the correct Bazel version for XLA. If Bazelisk is unavailable, you can
install Bazel manually.

We recommend using a
[TensorFlow Docker container](https://www.tensorflow.org/install/docker) to
build and test XLA.

To get the TensorFlow Docker image for CPU, run the following command:

```sh
docker run --name xla -w /xla -it -d --rm -v $PWD:/xla tensorflow/build:latest-python3.9 bash
```

To get the TensorFlow Docker image for GPU, run the following command:

```sh
docker run --name xla_gpu -w /xla -it -d --rm -v $PWD:/xla tensorflow/tensorflow:devel-gpu bash
```

## Build

Build for CPU:

```sh
docker exec xla ./configure
docker exec xla bazel build --test_output=all --spawn_strategy=sandboxed --nocheck_visibility //xla/...
```

Build for GPU:

```sh
docker exec -e TF_NEED_CUDA=1 xla_gpu ./configure
docker exec xla_gpu bazel build --test_output=all --spawn_strategy=sandboxed --nocheck_visibility //xla/...
```

For more information on building XLA, see
[Build from source](build_from_source.md).

## Create a pull request

When you're ready to send changes for review, create a
[pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

To learn about the XLA code review philosophy, see
[Code reviews](code_reviews.md).
