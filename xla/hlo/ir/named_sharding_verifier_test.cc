/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/ir/named_sharding_verifier.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

using DimensionSharding = NamedSharding::DimensionSharding;

TEST(NamedShardingVerifierTest, ValidSharding) {
  Mesh mesh({2, 4}, {"a", "b"});
  AxisRef a(0);
  AxisRef b(1);

  DimensionSharding ds_a({a}, /*is_closed=*/true);
  NamedSharding sharding(mesh, {ds_a}, {b});

  TF_EXPECT_OK(VerifyNamedSharding(sharding));
}

TEST(NamedShardingVerifierTest, ValidShardingWithSubAxes) {
  Mesh mesh({4}, {"a"});
  AxisRef a1(0, {1, 2});
  AxisRef a2(0, {2, 2});

  DimensionSharding ds_a1({a1}, /*is_closed=*/true);
  NamedSharding sharding(mesh, {ds_a1}, {a2});

  TF_EXPECT_OK(VerifyNamedSharding(sharding));
}

TEST(NamedShardingVerifierTest, InvalidAxisIndex) {
  Mesh mesh({2}, {"a"});
  AxisRef b(1);  // Index 1 is out of bounds for size 1

  DimensionSharding ds_b({b}, /*is_closed=*/true);

  EXPECT_EQ(VerifyNamedSharding(mesh, {ds_b}, {}, {}).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(NamedShardingVerifierTest, OverlappingAxesSameDim) {
  Mesh mesh({2}, {"a"});
  AxisRef a(0);

  DimensionSharding ds_aa({a, a}, /*is_closed=*/true);

  EXPECT_EQ(VerifyNamedSharding(mesh, {ds_aa}, {}, {}).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(NamedShardingVerifierTest, OverlappingAxesDifferentDims) {
  Mesh mesh({2}, {"a"});
  AxisRef a(0);

  DimensionSharding ds_a({a}, /*is_closed=*/true);

  EXPECT_EQ(VerifyNamedSharding(mesh, {ds_a, ds_a}, {}, {}).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(NamedShardingVerifierTest, OverlappingAxesDimAndReplicated) {
  Mesh mesh({2}, {"a"});
  AxisRef a(0);

  DimensionSharding ds_a({a}, /*is_closed=*/true);

  EXPECT_EQ(VerifyNamedSharding(mesh, {ds_a}, {a}, {}).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(NamedShardingVerifierTest, OverlappingAxesDimAndUnreduced) {
  Mesh mesh({2}, {"a"});
  AxisRef a(0);

  DimensionSharding ds_a({a}, /*is_closed=*/true);

  EXPECT_EQ(VerifyNamedSharding(mesh, {ds_a}, {}, {a}).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(NamedShardingVerifierTest, OverlappingSubAxes) {
  Mesh mesh({4}, {"a"});
  AxisRef whole(0, {1, 4});
  AxisRef half(0, {1, 2});

  DimensionSharding ds({whole}, /*is_closed=*/true);

  EXPECT_EQ(VerifyNamedSharding(mesh, {ds}, {half}, {}).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace xla
