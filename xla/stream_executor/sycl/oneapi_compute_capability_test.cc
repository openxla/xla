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

TEST(OneAPIComputeCapabilityTest, ProtoTest1) {
  OneAPIComputeCapabilityProto proto;
  proto.set_architecture("BMG");
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability(proto);
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
  EXPECT_FALSE(compute_capability.IsMTL());
  EXPECT_FALSE(compute_capability.IsARL());
  EXPECT_TRUE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsLNL());
  EXPECT_FALSE(compute_capability.IsPTL());
  EXPECT_FALSE(compute_capability.IsWCL());
  EXPECT_FALSE(compute_capability.IsNVL());
  EXPECT_FALSE(compute_capability.IsNVLP());
}

TEST(OneAPIComputeCapabilityTest, ProtoTest2) {
  OneAPIComputeCapabilityProto proto;
  proto.set_architecture("DG2");
  proto.set_variant("G11");
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability(proto);
  EXPECT_TRUE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
  EXPECT_FALSE(compute_capability.IsMTL());
  EXPECT_FALSE(compute_capability.IsARL());
  EXPECT_FALSE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsLNL());
  EXPECT_FALSE(compute_capability.IsPTL());
  EXPECT_FALSE(compute_capability.IsWCL());
  EXPECT_FALSE(compute_capability.IsNVL());
  EXPECT_FALSE(compute_capability.IsNVLP());
}

TEST(OneAPIComputeCapabilityTest, ComputeCapabilityFromName) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("pvc", "vg");
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_TRUE(compute_capability.IsPVC());
  EXPECT_FALSE(compute_capability.IsMTL());
  EXPECT_FALSE(compute_capability.IsARL());
  EXPECT_FALSE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsLNL());
  EXPECT_FALSE(compute_capability.IsPTL());
  EXPECT_FALSE(compute_capability.IsWCL());
  EXPECT_FALSE(compute_capability.IsNVL());
  EXPECT_FALSE(compute_capability.IsNVLP());
}

TEST(OneAPIComputeCapabilityTest, ComputeCapabilityFromGenAndVer) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability(0x14, 0x2);
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
  EXPECT_FALSE(compute_capability.IsMTL());
  EXPECT_FALSE(compute_capability.IsARL());
  EXPECT_TRUE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsLNL());
  EXPECT_FALSE(compute_capability.IsPTL());
  EXPECT_FALSE(compute_capability.IsWCL());
  EXPECT_FALSE(compute_capability.IsNVL());
  EXPECT_FALSE(compute_capability.IsNVLP());
}

TEST(OneAPIComputeCapabilityTest, ComputeCapabilityForUnknownDevice) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("Device", "Variant");
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
  EXPECT_FALSE(compute_capability.IsMTL());
  EXPECT_FALSE(compute_capability.IsARL());
  EXPECT_FALSE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsLNL());
  EXPECT_FALSE(compute_capability.IsPTL());
  EXPECT_FALSE(compute_capability.IsWCL());
  EXPECT_FALSE(compute_capability.IsNVL());
  EXPECT_FALSE(compute_capability.IsNVLP());
  EXPECT_EQ(compute_capability.ToString(), "Unknown");
}

TEST(OneAPIComputeCapabilityTest, ComputeCapabilityForUnknownVariant) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("BMG", "Variant");
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
  EXPECT_FALSE(compute_capability.IsMTL());
  EXPECT_FALSE(compute_capability.IsARL());
  EXPECT_FALSE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsLNL());
  EXPECT_FALSE(compute_capability.IsPTL());
  EXPECT_FALSE(compute_capability.IsWCL());
  EXPECT_FALSE(compute_capability.IsNVL());
  EXPECT_FALSE(compute_capability.IsNVLP());
  EXPECT_EQ(compute_capability.ToString(), "Xe2");
}

TEST(OneAPIComputeCapabilityTest, ToString) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("DG2", "G10");
  EXPECT_EQ(compute_capability.ToString(), "DG2_G10");
}

TEST(OneAPIComputeCapabilityTest, ToProto) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("PVC", "VG");
  OneAPIComputeCapabilityProto proto = compute_capability.ToProto();
  EXPECT_EQ(proto.architecture(), "PVC");
  EXPECT_EQ(proto.variant(), "VG");
}

TEST(OneAPIComputeCapabilityTest, ToProtoUnknownDevice) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("ABC", "DEF");
  OneAPIComputeCapabilityProto proto = compute_capability.ToProto();
  EXPECT_EQ(proto.architecture(), "Unknown");
  EXPECT_EQ(proto.variant(), "");
}

TEST(OneAPIComputeCapabilityTest, UtilityFunc) {
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability::BMG();
  EXPECT_FALSE(compute_capability.IsDG2());
  EXPECT_FALSE(compute_capability.IsPVC());
  EXPECT_FALSE(compute_capability.IsMTL());
  EXPECT_FALSE(compute_capability.IsARL());
  EXPECT_TRUE(compute_capability.IsBMG());
  EXPECT_FALSE(compute_capability.IsLNL());
  EXPECT_FALSE(compute_capability.IsPTL());
  EXPECT_FALSE(compute_capability.IsWCL());
  EXPECT_FALSE(compute_capability.IsNVL());
  EXPECT_FALSE(compute_capability.IsNVLP());
}

TEST(OneAPIComputeCapabilityTest, GetterFunc) {
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability::DG2();
  EXPECT_EQ(compute_capability.architecture(), "DG2");
  EXPECT_EQ(compute_capability.variant(), "G10");
}

TEST(OneAPIComputeCapabilityTest, RoundTrip1) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("MTL", "U");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTrip2) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("MTL", "H");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTrip3) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("ARL", "H");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTrip4) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("LNL", "M");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTrip5) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("PTL", "H");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTrip6) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("PTL", "U");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTrip7) {
  OneAPIComputeCapability compute_capability = OneAPIComputeCapability("WCL");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTrip8) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("NVL", "S");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTrip9) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("NVL", "U");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTrip10) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("NVLP", "P");
  EXPECT_TRUE(OneAPIComputeCapability::FromProto(
                  compute_capability.ToProto()) == compute_capability);
}

TEST(OneAPIComputeCapabilityTest, RoundTripUnknown) {
  OneAPIComputeCapability compute_capability =
      OneAPIComputeCapability("BMG", "Variant");

  EXPECT_FALSE(OneAPIComputeCapability::FromProto(
                   compute_capability.ToProto()) == compute_capability);
}

}  // namespace
}  // namespace stream_executor::sycl
