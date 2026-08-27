# XLA development container

This directory provides a small wrapper around the same `ml-build` image used
by XLA CI. The repository is mounted at `/workspace/xla`, while Bazel caches are
kept under `${HOME}/.cache/xla-docker` on the host.

Start an interactive development shell:

```bash
build_tools/dev_container/run.sh
```

Run the first HLO pass test directly:

```bash
build_tools/dev_container/run.sh \
  bazel test --config=clang_local \
  //xla/examples/first_hlo_pass:first_hlo_pass_test \
  --test_output=errors
```

To expose NVIDIA GPUs to the container, set `XLA_DOCKER_GPUS`:

```bash
XLA_DOCKER_GPUS=all build_tools/dev_container/run.sh
```

Override the container image, its base image, or the cache location with
`XLA_DEV_IMAGE`, `XLA_BUILD_IMAGE`, and `XLA_DOCKER_CACHE`, respectively. If a
custom base image does not contain Bazel, set `XLA_BAZEL_BIN` to a host Bazel
8.7.0 executable; the wrapper mounts it at `/usr/local/bin/bazel`.
