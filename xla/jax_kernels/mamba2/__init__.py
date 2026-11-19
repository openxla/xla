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

"""Plain JAX mamba2 champion kernel package."""

from xla.jax_kernels.mamba2 import champion
from xla.jax_kernels.mamba2 import configs
from xla.jax_kernels.mamba2 import reference

baseline = reference
segsum = reference.segsum
ssd_reference = reference.ssd_reference
ssd_ref = reference.ssd_ref
ssd_grid = champion.ssd_grid
ssd_jax_k = champion.ssd_jax_k
Mamba2BenchmarkConfig = configs.Mamba2BenchmarkConfig
Mamba2ShapeConfig = configs.Mamba2ShapeConfig
MAMBA2_BENCHMARK_CONFIGS = configs.MAMBA2_BENCHMARK_CONFIGS
make_ssd_inputs = configs.make_ssd_inputs

__all__ = [
    "MAMBA2_BENCHMARK_CONFIGS",
    "Mamba2BenchmarkConfig",
    "Mamba2ShapeConfig",
    "baseline",
    "champion",
    "configs",
    "make_ssd_inputs",
    "reference",
    "segsum",
    "ssd_grid",
    "ssd_jax_k",
    "ssd_ref",
    "ssd_reference",
]
