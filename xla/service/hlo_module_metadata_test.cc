/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_module_metadata.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/metrics.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::Property;
using ::testing::StrEq;

class TestEnv : public tsl::EnvWrapper {
 public:
  TestEnv() : EnvWrapper(Env::Default()) {}

  uint64_t NowMicros() const override { return current_micros_; }

  void SetCurrentMicros(uint64_t micros) { current_micros_ = micros; }

 private:
  uint64_t current_micros_ = 1;
};

TEST(HloModuleMetadata, RecordsPassStart) {
  TestEnv env;
  HloModuleMetadata module_metadata(&env);
  env.SetCurrentMicros(1234);
  module_metadata.RecordPassStart();
  EXPECT_THAT(
      module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::start_timestamp_usec, 1234)));
}

TEST(HloModuleMetadata, RecordsPassEnd) {
  TestEnv env;
  HloModuleMetadata module_metadata(&env);
  module_metadata.RecordPassStart();
  env.SetCurrentMicros(4321);
  EXPECT_IS_OK(module_metadata.RecordPassEnd());
  EXPECT_THAT(
      module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::end_timestamp_usec, 4321)));
}

TEST(HloModuleMetadata, RecordsPassEndInNestedMetadata) {
  TestEnv env;
  HloModuleMetadata module_metadata(&env);
  module_metadata.RecordPassStart();
  module_metadata.RecordPassStart();
  env.SetCurrentMicros(111);
  EXPECT_IS_OK(module_metadata.RecordPassEnd());
  EXPECT_THAT(module_metadata.proto().pass_metadata(),
              ElementsAre(Property(&HloPassMetadata::end_timestamp_usec, 0),
                          Property(&HloPassMetadata::end_timestamp_usec, 111)));

  env.SetCurrentMicros(222);
  EXPECT_IS_OK(module_metadata.RecordPassEnd());
  EXPECT_THAT(module_metadata.proto().pass_metadata(),
              ElementsAre(Property(&HloPassMetadata::end_timestamp_usec, 222),
                          Property(&HloPassMetadata::end_timestamp_usec, 111)));
}

TEST(HloModuleMetadata, RecordPassEndReturnsNotFound) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  EXPECT_EQ(module_metadata.RecordPassEnd().code(), tsl::error::NOT_FOUND);

  module_metadata.RecordPassStart();
  EXPECT_IS_OK(module_metadata.RecordPassEnd());
  EXPECT_EQ(module_metadata.RecordPassEnd().code(), tsl::error::NOT_FOUND);
}

TEST(HloModuleMetadata, SetsHloPassMetadataFields) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  module_metadata.RecordPassStart();
  EXPECT_IS_OK(module_metadata.set_current_pass_name("fake name"));
  EXPECT_THAT(
      module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::pass_name, StrEq("fake name"))));
}

TEST(HloModuleMetadata, SetsHloPassMetadataFieldsInNestedMetadata) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  module_metadata.RecordPassStart();
  module_metadata.RecordPassStart();
  EXPECT_IS_OK(module_metadata.set_current_pass_name("fake name"));
  EXPECT_THAT(
      module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::pass_name, StrEq("")),
                  Property(&HloPassMetadata::pass_name, StrEq("fake name"))));
}

TEST(HloModuleMetadata, SetterReturnsNotFound) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  EXPECT_EQ(module_metadata.set_current_pass_name("fake name").code(),
            tsl::error::NOT_FOUND);
}

TEST(HloModuleMetadata, CopiesRunningPrepartitioningPasses) {
  HloModuleMetadata old_module_metadata(tsl::Env::Default());
  old_module_metadata.RecordPassStart();
  EXPECT_IS_OK(old_module_metadata.set_current_pass_name("outer pass"));

  old_module_metadata.RecordPassStart();
  EXPECT_IS_OK(old_module_metadata.set_current_pass_name("finished pass"));
  EXPECT_IS_OK(old_module_metadata.RecordPassEnd());

  old_module_metadata.RecordPassStart();
  EXPECT_IS_OK(old_module_metadata.set_current_pass_name("inner pass"));

  HloModuleMetadata new_module_metadata(tsl::Env::Default());
  new_module_metadata.set_prepartitioning_metadata(old_module_metadata);

  // Passes that are still running go in the new module.
  EXPECT_THAT(
      new_module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::pass_name, StrEq("outer pass")),
                  Property(&HloPassMetadata::pass_name, StrEq("inner pass"))));

  // Passes that finished go in the prepartitioning metadata.
  EXPECT_THAT(new_module_metadata.prepartitioning_metadata()->pass_metadata(),
              ElementsAre(Property(&HloPassMetadata::pass_name,
                                   StrEq("finished pass"))));
}

KeyValueMetric DecisionPayload(int64_t value) {
  KeyValueMetric payload;
  payload.set_key("decision");
  payload.set_value(value);
  return payload;
}

TEST(HloModuleMetadata, AddsDecisionRecordForCurrentPass) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  module_metadata.RecordPassStart();
  EXPECT_IS_OK(module_metadata.set_current_pass_name("outer pass"));
  module_metadata.RecordPassStart();
  EXPECT_IS_OK(module_metadata.set_current_pass_name("inner pass"));
  EXPECT_IS_OK(module_metadata.set_current_pass_pipeline_name("pipeline"));
  ASSERT_OK_AND_ASSIGN(int64_t inner_pass_id,
                       module_metadata.current_pass_id());

  EXPECT_IS_OK(module_metadata.AddDecisionRecord(DecisionPayload(1)));
  EXPECT_IS_OK(module_metadata.AddDecisionRecord(DecisionPayload(2)));

  ASSERT_EQ(module_metadata.decision_records().size(), 2);
  for (int i = 0; i < 2; ++i) {
    const HloPassDecisionRecord& record = module_metadata.decision_records()[i];
    EXPECT_EQ(record.pass_id(), inner_pass_id);
    EXPECT_EQ(record.pass_name(), "inner pass");
    EXPECT_EQ(record.pipeline_name(), "pipeline");
    KeyValueMetric payload;
    ASSERT_TRUE(record.payload().UnpackTo(&payload));
    EXPECT_EQ(payload.value(), i + 1);
  }
}

TEST(HloModuleMetadata, AddDecisionRecordReturnsNotFoundOutsidePass) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  EXPECT_EQ(module_metadata.AddDecisionRecord(DecisionPayload(1)).code(),
            tsl::error::NOT_FOUND);
  module_metadata.RecordPassStart();
  EXPECT_IS_OK(module_metadata.RecordPassEnd());
  EXPECT_EQ(module_metadata.AddDecisionRecord(DecisionPayload(1)).code(),
            tsl::error::NOT_FOUND);
  EXPECT_TRUE(module_metadata.decision_records().empty());
}

TEST(HloModuleMetadata, DecisionRecordsSurviveCloneAndMove) {
  HloModule module("m", HloModuleConfig());
  module.metadata()->RecordPassStart();
  EXPECT_IS_OK(module.metadata()->set_current_pass_name("pass"));
  EXPECT_IS_OK(module.metadata()->AddDecisionRecord(DecisionPayload(7)));
  EXPECT_IS_OK(module.metadata()->RecordPassEnd());

  std::unique_ptr<const HloModule> clone = module.Clone();
  const HloModuleMetadata& cloned = clone->metadata();
  ASSERT_EQ(cloned.decision_records().size(), 1);
  EXPECT_EQ(cloned.decision_records()[0].pass_name(), "pass");

  HloModule destination("destination", HloModuleConfig());
  std::unique_ptr<HloModule> source = module.Clone();
  source->MoveMetadataToModule(&destination);
  const HloModuleMetadata& moved = std::as_const(destination).metadata();
  ASSERT_EQ(moved.decision_records().size(), 1);
  KeyValueMetric payload;
  ASSERT_TRUE(moved.decision_records()[0].payload().UnpackTo(&payload));
  EXPECT_EQ(payload.value(), 7);
}

}  // namespace
}  // namespace xla
