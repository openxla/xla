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

#include <gtest/gtest.h>
#include "third_party/protobuf/descriptor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

TEST(DebugOptionsTest, AllFieldsHaveExplicitPresence) {
  for (int i = 0; i < DebugOptions::descriptor()->field_count(); ++i) {
    const proto2::FieldDescriptor* field = DebugOptions::descriptor()->field(i);
    EXPECT_TRUE(field->is_map() || field->is_repeated() ||
                field->has_presence())
        << "DebugOptions field " << field->name()
        << " uses implicit presence, which makes merging DebugOptions "
           "instances hard when derived from flags where the flag default and "
           "proto default may differ. In most cases, the field should "
           "explicitly be marked as 'optional' or 'repeated'. Maps are a "
           "special case of repeated fields.";
  }
}

}  // namespace
}  // namespace xla
