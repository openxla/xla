/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/codegen/emitters/transforms/nontemporal_hint_targets.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_description.h"

namespace xla::emitters {
namespace {

stream_executor::DeviceDescription RocmDevice(absl::string_view gcn_arch_name) {
  stream_executor::DeviceDescription d;
  d.set_rocm_compute_capability(std::string(gcn_arch_name));
  return d;
}

TEST(NontemporalHintTargets, PerTargetSelection) {
  // gfx942: loads and stores; single-output stores; no loads when overwriting.
  auto gfx942 = GetNontemporalHintTargets(RocmDevice("gfx942:sramecc+:xnack-"));
  EXPECT_TRUE(gfx942.loads);
  EXPECT_TRUE(gfx942.stores);
  EXPECT_TRUE(gfx942.single_output_stores);
  EXPECT_TRUE(gfx942.out_of_place_loads);
  EXPECT_EQ(gfx942.stream_l2_multiple, 16);

  // Other targets get no hints.
  auto other = GetNontemporalHintTargets(RocmDevice("gfx90a:sramecc+:xnack"));
  EXPECT_FALSE(other.loads);
  EXPECT_FALSE(other.stores);
}

}  // namespace
}  // namespace xla::emitters
