/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/python/idempotent_serializer.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/sharding.pb.h"

namespace xla {
namespace {

constexpr absl::string_view kSerializedDummy = "dummy";

TEST(IdempotentSerializerTest, Test) {
  EXPECT_FALSE(IdempotentSerializer::is_registered());
  EXPECT_TRUE(IdempotentSerializer::Register(
      [](const tsl::protobuf::Message&) -> std::string {
        return std::string(kSerializedDummy);
      }));
  EXPECT_TRUE(IdempotentSerializer::is_registered());
  xla::ifrt::SingleDeviceShardingProto dummy;
  EXPECT_EQ(IdempotentSerializer::Serialize(dummy), kSerializedDummy);
}

}  // namespace
}  // namespace xla
