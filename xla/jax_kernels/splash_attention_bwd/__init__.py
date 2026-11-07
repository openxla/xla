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

"""Plain JAX splash_attention_bwd champion kernel package."""

from xla.jax_kernels.splash_attention_bwd import champion
from xla.jax_kernels.splash_attention_bwd import configs
from xla.jax_kernels.splash_attention_bwd import reference

baseline = reference
splash_attention_bwd_jax_k = champion.splash_attention_bwd_jax_k
splash_attention_mha_bwd_jax_k = champion.splash_attention_mha_bwd_jax_k
splash_attention_gqa_bwd_jax_k = champion.splash_attention_gqa_bwd_jax_k
SplashBwdJaxKConfig = champion.SplashBwdJaxKConfig
CANONICAL_BENCHMARK_CONFIGS = configs.CANONICAL_BENCHMARK_CONFIGS
DEFAULT_MASK_VALUE = configs.DEFAULT_MASK_VALUE
LOG2E = configs.LOG2E
LOG2E_INV = configs.LOG2E_INV
LN2 = configs.LN2
SplashBwdConfig = configs.SplashBwdConfig
reference_splash_attention_fwd = reference.reference_splash_attention_fwd
reference_splash_attention_bwd = reference.reference_splash_attention_bwd
CausalMask = reference.CausalMask
ChunkedCausalMask = reference.ChunkedCausalMask
FullMask = reference.FullMask
LocalMask = reference.LocalMask
Mask = reference.Mask
MultiHeadMask = reference.MultiHeadMask
NumpyMask = reference.NumpyMask

__all__ = [
    "CANONICAL_BENCHMARK_CONFIGS",
    "CausalMask",
    "ChunkedCausalMask",
    "DEFAULT_MASK_VALUE",
    "FullMask",
    "LN2",
    "LOG2E",
    "LOG2E_INV",
    "LocalMask",
    "Mask",
    "MultiHeadMask",
    "NumpyMask",
    "SplashBwdConfig",
    "SplashBwdJaxKConfig",
    "baseline",
    "champion",
    "configs",
    "reference",
    "reference_splash_attention_bwd",
    "reference_splash_attention_fwd",
    "splash_attention_bwd_jax_k",
    "splash_attention_gqa_bwd_jax_k",
    "splash_attention_mha_bwd_jax_k",
]
