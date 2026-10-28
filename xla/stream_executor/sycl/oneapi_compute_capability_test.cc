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

#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

#include <gtest/gtest.h>
#include "xla/stream_executor/sycl/oneapi_compute_capability.pb.h"

namespace stream_executor::sycl {
namespace {

TEST(OneAPIComputeCapabilityTest, FromProtoTest) {
  OneAPIComputeCapabilityProto proto;
  proto.set_architecture("Xe2HPG");
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability(proto);
  EXPECT_TRUE(compute_capability.IsXe2HPG());
  EXPECT_FALSE(compute_capability.IsXeHPG());
  EXPECT_FALSE(compute_capability.IsXeHPC());
}

TEST(OneAPIComputeCapabilityTest, ToProtoTest) {
  OneAPIComputeCapability pvc_cc = OneAPIComputeCapability::PVC();
  OneAPIComputeCapabilityProto proto = pvc_cc.ToProto();
  EXPECT_EQ(proto.architecture(), "XeHPC");
}

TEST(OneAPIComputeCapabilityTest, ToStringTest) {
  OneAPIComputeCapability nvlp_cc = OneAPIComputeCapability("Xe3PLPG");
  EXPECT_EQ(nvlp_cc.ToString(), "Xe3PLPG");
}

TEST(OneAPIComputeCapabilityTest, ConstructFromArchNameTest) {
  EXPECT_TRUE(OneAPIComputeCapability("xehpc").IsXeHPC());
  EXPECT_TRUE(OneAPIComputeCapability("XEHPG").IsXeHPG());
  EXPECT_EQ(OneAPIComputeCapability("ABC").ToString(), "Unknown");
}

TEST(OneAPIComputeCapabilityTest, ConstructFromGenAndVerTest) {
  EXPECT_TRUE(OneAPIComputeCapability(XeGeneration::kXe, 60).IsXeHPC());
  EXPECT_TRUE(OneAPIComputeCapability(XeGeneration::kXe, 56).IsXeHPG());
  EXPECT_EQ(OneAPIComputeCapability(XeGeneration::kXe, 0xff).ToString(),
            "XeOther");
  EXPECT_EQ(OneAPIComputeCapability(XeGeneration::kUnknown, 20).ToString(),
            "Unknown");
}

TEST(OneAPIComputeCapabilityTest, NamedDevicePrecedenceOverFallbackTest) {
  OneAPIComputeCapability bmg_cc(XeGeneration::kXe2, 2);
  EXPECT_TRUE(bmg_cc.IsXe2HPG());
  EXPECT_FALSE(bmg_cc.IsXe2Other());
}

TEST(OneAPIComputeCapabilityTest, ExtendedVersionRangeForArchTest) {
  OneAPIComputeCapability adl_cc(XeGeneration::kXe, 0);
  OneAPIComputeCapability dg1_cc(XeGeneration::kXe, 10);
  EXPECT_TRUE(adl_cc.IsXeLP());
  EXPECT_TRUE(dg1_cc.IsXeLP());
  EXPECT_EQ(adl_cc, dg1_cc);
}

TEST(OneAPIComputeCapabilityTest, UnlistedVersionsForArchTest) {
  EXPECT_TRUE(OneAPIComputeCapability(XeGeneration::kXe, 99).IsXeOther());
  EXPECT_TRUE(OneAPIComputeCapability(XeGeneration::kXe2, 7).IsXe2Other());
  EXPECT_TRUE(OneAPIComputeCapability(XeGeneration::kXe3P, 0).IsXe3POther());
  EXPECT_FALSE(OneAPIComputeCapability(XeGeneration::kXe2, 4).IsXe2Other());
}

TEST(OneAPIComputeCapabilityTest, UnknownDeviceTest) {
  OneAPIComputeCapability unknown_cc = OneAPIComputeCapability();
  EXPECT_EQ(unknown_cc.ToString(), "Unknown");
}

TEST(OneAPIComputeCapabilityTest, UnknownDeviceToProtoTest) {
  OneAPIComputeCapability unknown_cc = OneAPIComputeCapability();
  OneAPIComputeCapabilityProto proto = unknown_cc.ToProto();
  EXPECT_EQ(proto.architecture(), "Unknown");
}

TEST(OneAPIComputeCapabilityTest, NamedUtilityFunctionsTest) {
  EXPECT_TRUE(OneAPIComputeCapability::BMG().IsXe2HPG());
  EXPECT_TRUE(OneAPIComputeCapability::DG2().IsXeHPG());
  EXPECT_TRUE(OneAPIComputeCapability::PVC().IsXeHPC());
  EXPECT_NE(OneAPIComputeCapability::BMG(), OneAPIComputeCapability::PVC());
}

TEST(OneAPIComputeCapabilityTest, ProtoRoundTripTest) {
  OneAPIComputeCapability original =
      OneAPIComputeCapability(XeGeneration::kXe2, 4);
  OneAPIComputeCapabilityProto proto = original.ToProto();
  EXPECT_EQ(proto.architecture(), "Xe2LPG");
  EXPECT_EQ(OneAPIComputeCapability::FromProto(proto), original);
}

}  // namespace
}  // namespace stream_executor::sycl
