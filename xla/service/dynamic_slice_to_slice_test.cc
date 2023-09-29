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

#include "xla/service/dynamic_slice_to_slice.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class DynamicSliceToSliceTest : public HloTestBase {};

TEST_F(DynamicSliceToSliceTest, ReplaceSimpleDynamicSlice) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    ENTRY entry {
      %p0 = bf16[64,2048,768] parameter(0)
      %p1 = s32[] constant(1)
      %i0 = s32[] constant(0)
      %dys = bf16[32,20,768] dynamic-slice(%p0, %p1, %i0, %i0), dynamic_slice_sizes={32,20,768}
      ROOT out = bf16[32,20,768] add(%dys, %dys)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  ASSERT_TRUE(DynamicSliceToSlice().Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Add(match::Slice(match::Parameter(0)),
                                    match::Slice(match::Parameter(0)))));
}

TEST_F(DynamicSliceToSliceTest, ReplaceSimpleDynamicSliceWithLayout) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    ENTRY entry {
      %p0 = bf16[64,2048,768] parameter(0)
      %p1 = s32[] constant(1)
      %i0 = s32[] constant(0)
      %dys = bf16[32,20,768]{2,1,0:T(8,128)} dynamic-slice(%p0, %p1, %i0, %i0), dynamic_slice_sizes={32,20,768}
      ROOT out = bf16[32,20,768]{2,1,0:T(8,128)} add(%dys, %dys)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  ASSERT_TRUE(DynamicSliceToSlice().Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Add(match::Slice(match::Parameter(0)),
                                    match::Slice(match::Parameter(0)))));
}

TEST_F(DynamicSliceToSliceTest, ReplaceSimpleDynamicSliceOutOfBound) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    ENTRY entry {
      %p0 = bf16[64,2048,768] parameter(0)
      %p1 = s32[] constant(1)
      %i0 = s32[] constant(0)
      %dys = bf16[64,2048,768] dynamic-slice(%p0, %p1, %i0, %i0), dynamic_slice_sizes={64,2048,768}
      ROOT out = bf16[64,2048,768] add(%dys, %dys)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  ASSERT_TRUE(DynamicSliceToSlice().Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Add(match::Slice(match::Parameter(0)),
                                    match::Slice(match::Parameter(0)))));
}

TEST_F(DynamicSliceToSliceTest, NotReplaceSimpleDynamicSlice) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    ENTRY entry {
      %p0 = bf16[64,2048,768] parameter(0)
      %p1 = s32[] parameter(1)
      %i0 = s32[] constant(0)
      %dys = bf16[32,2048,768] dynamic-slice(%p0, %p1, %i0, %i0), dynamic_slice_sizes={32,2048,768}
      ROOT out = bf16[32,2048,768] add(%dys, %dys)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  EXPECT_FALSE(DynamicSliceToSlice().Run(module.get()).value());
}

TEST_F(DynamicSliceToSliceTest, ReplaceRootDynamicSlice) {
  const char* const kHloModule = R"(
    HloModule ModuleWithWhile

    ENTRY entry {
      %p0 = bf16[64,2048,768] parameter(0)
      %p1 = s32[] constant(1)
      %i0 = s32[] constant(0)
      ROOT %dys = bf16[32,20,768] dynamic-slice(%p0, %p1, %i0, %i0), dynamic_slice_sizes={32,20,768}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  ASSERT_TRUE(DynamicSliceToSlice().Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Slice(match::Parameter(0))));
}

}  // namespace
}  // namespace xla
