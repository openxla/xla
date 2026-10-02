# Copyright 2026 The OpenXLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import enum

class GpuTargetConfig:
  @property
  def platform_name(self) -> str: ...
  @property
  def device_description_str(self) -> str: ...
  @property
  def arch_name(self) -> str: ...
  @property
  def compute_capability(self) -> int: ...
  @property
  def core_count(self) -> int: ...
  @property
  def smem_capacity_bytes(self) -> int: ...

class GpuModel(enum.Enum):
  A100_PCIE_80 = "A100_PCIE_80"
  A100_SXM_40 = "A100_SXM_40"
  A100_SXM_80 = "A100_SXM_80"
  A6000 = "A6000"
  B200 = "B200"
  B300 = "B300"
  BMG_G21 = "BMG_G21",
  H100_PCIE = "H100_PCIE"
  H100_SXM = "H100_SXM"
  H200 = "H200"
  MI200 = "MI200"
  P100 = "P100"
  PVC = "PVC"
  V100 = "V100"
  GB200 = "GB200"
  GB300 = "GB300"
  RTX6000PRO = "RTX6000PRO"
  GFX1250 = "GFX1250"
  MI350 = "MI350"

def get_gpu_spec(gpu_model: str) -> GpuTargetConfig: ...
