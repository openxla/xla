#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${XLA_DEV_IMAGE:-xla-dev:latest}"
build_image="${XLA_BUILD_IMAGE:-us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest}"
cache_dir="${XLA_DOCKER_CACHE:-${HOME}/.cache/xla-docker}"

mkdir -p "${cache_dir}"

docker build \
  --build-arg "XLA_BUILD_IMAGE=${build_image}" \
  --tag "${image}" \
  --file "${repo_root}/build_tools/dev_container/Dockerfile" \
  "${repo_root}/build_tools/dev_container"

docker_args=(
  run
  --rm
  --user "$(id -u):$(id -g)"
  --env HOME=/home/xla
  --env USER=xla
  --volume "${repo_root}:/workspace/xla"
  --volume "${cache_dir}:/home/xla"
  --workdir /workspace/xla
)

if [[ -n "${XLA_BAZEL_BIN:-}" ]]; then
  docker_args+=(--volume "${XLA_BAZEL_BIN}:/usr/local/bin/bazel:ro")
fi

if [[ -t 0 && -t 1 ]]; then
  docker_args+=(-it)
fi

if [[ -n "${XLA_DOCKER_GPUS:-}" ]]; then
  docker_args+=(--gpus "${XLA_DOCKER_GPUS}")
fi

docker_args+=("${image}")

if [[ $# -gt 0 ]]; then
  docker_args+=("$@")
fi

exec docker "${docker_args[@]}"
