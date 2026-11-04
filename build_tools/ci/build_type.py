# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Enum of XLA and JAX CI build configurations."""

from __future__ import annotations

import enum


class BuildType(enum.Enum):
  """Enum representing all types of builds.

  Should be named as `REPO,OS,HOST_TYPE,BACKEND,GPU_TYPE,CI_TYPE`.
  """

  XLA_LINUX_X86_CPU_GITHUB_ACTIONS = enum.auto()
  XLA_WINDOWS_X86_CPU_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_CPU_WORKSPACE_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_ARM64_CPU_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_GPU_T4_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_GPU_8X_H100_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_GPU_ONEAPI_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_GPU_HERMETIC_ROCM_GITHUB_ACTIONS = enum.auto()

  # Presubmit builds for regression testing.
  XLA_LINUX_ARM64_CPU_48_VCPU_PRESUBMIT_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_CPU_128_VCPU_PRESUBMIT_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_GPU_L4_16_VCPU_BENCHMARK_PRESUBMIT_GITHUB_ACTIONS = enum.auto()
  XLA_LINUX_X86_GPU_A4_224_VCPU_BENCHMARK_PRESUBMIT_GITHUB_ACTIONS = enum.auto()

  XLA_MACOS_X86_CPU_KOKORO = enum.auto()
  XLA_MACOS_ARM64_CPU_KOKORO = enum.auto()

  JAX_LINUX_X86_CPU_BZLMOD_GITHUB_ACTIONS = enum.auto()
  JAX_WINDOWS_X86_CPU_GITHUB_ACTIONS = enum.auto()
  JAX_LINUX_X86_GPU_T4_GITHUB_ACTIONS = enum.auto()

  @classmethod
  def from_str(cls, build_name: str) -> BuildType:
    """Returns the BuildType matching build_name (case- and space-insensitive).

    Args:
      build_name: Name of the build configuration.

    Raises:
      ValueError: If build_name does not match any BuildType member.
    """
    try:
      return cls[build_name.replace(" ", "_").upper()]
    except KeyError as e:
      raise ValueError(f"Unknown build type: {build_name!r}") from e


def is_jax_cpu_build(build_type: BuildType | None) -> bool:
  """Returns True if build_type is a CPU JAX build."""
  return build_type in (
      BuildType.JAX_LINUX_X86_CPU_BZLMOD_GITHUB_ACTIONS,
      BuildType.JAX_WINDOWS_X86_CPU_GITHUB_ACTIONS,
  )
