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
  def shared_memory_per_core(self) -> int: ...

def get_gpu_spec(device_kind: str) -> GpuTargetConfig: ...
