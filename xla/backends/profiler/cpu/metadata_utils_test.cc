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

#include "xla/backends/profiler/cpu/metadata_utils.h"

#include <cstdint>
#include <string>

#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

using tsl::profiler::GetStatTypeStr;
using tsl::profiler::HloModuleNameWithProgramId;
using tsl::profiler::StatType;

xla::HloProto MakeHloProto(const std::string& module_name, uint64_t module_id) {
  xla::HloProto hlo_proto;
  hlo_proto.mutable_hlo_module()->set_name(module_name);
  hlo_proto.mutable_hlo_module()->set_id(module_id);
  return hlo_proto;
}

TEST(MetadataUtilsTest, MultipleModulesUseActualProgramIdsAsKeys) {
  tensorflow::profiler::XPlane plane;
  MetadataXPlaneBuilder builder(&plane);

  const uint64_t ids[] = {664863, 665699, 666536};
  for (uint64_t id : ids) {
    builder.AddHloProto(id, MakeHloProto("jit_fn", id));
  }

  ASSERT_EQ(plane.event_metadata().size(), 3);
  for (uint64_t id : ids) {
    EXPECT_TRUE(plane.event_metadata().contains(id))
        << "Missing event_metadata for program_id=" << id;
  }
  EXPECT_FALSE(plane.event_metadata().contains(1));
  EXPECT_FALSE(plane.event_metadata().contains(2));
  EXPECT_FALSE(plane.event_metadata().contains(3));
}

TEST(MetadataUtilsTest, AddHloProtoSetsNameAndDisplayName) {
  tensorflow::profiler::XPlane plane;
  MetadataXPlaneBuilder builder(&plane);

  const uint64_t program_id = 664863;
  const std::string module_name = "jit_fn";
  builder.AddHloProto(program_id, MakeHloProto(module_name, program_id));

  const auto& em = plane.event_metadata().at(program_id);
  const std::string expected_name =
      HloModuleNameWithProgramId(module_name, program_id);
  EXPECT_EQ(em.name(), expected_name);
  EXPECT_EQ(em.display_name(), expected_name);
}

TEST(MetadataUtilsTest, AddHloProtoStoresProgramIdStat) {
  tensorflow::profiler::XPlane plane;
  MetadataXPlaneBuilder builder(&plane);

  const uint64_t program_id = 664863;
  builder.AddHloProto(program_id, MakeHloProto("jit_fn", program_id));

  const std::string kProgramIdName(GetStatTypeStr(StatType::kProgramId));
  int64_t program_id_stat_key = -1;
  for (const auto& [key, sm] : plane.stat_metadata()) {
    if (sm.name() == kProgramIdName) {
      program_id_stat_key = key;
      break;
    }
  }
  ASSERT_GE(program_id_stat_key, 0) << "kProgramId stat metadata not found";

  const auto& em = plane.event_metadata().at(program_id);
  bool found = false;
  for (const auto& stat : em.stats()) {
    if (stat.metadata_id() == program_id_stat_key) {
      EXPECT_EQ(static_cast<uint64_t>(stat.int64_value()), program_id);
      found = true;
    }
  }
  EXPECT_TRUE(found) << "kProgramId stat not found on event_metadata";
}

TEST(MetadataUtilsTest, AddHloProtoIsIdempotent) {
  tensorflow::profiler::XPlane plane;
  MetadataXPlaneBuilder builder(&plane);

  const uint64_t program_id = 664863;
  builder.AddHloProto(program_id, MakeHloProto("jit_fn", program_id));
  builder.AddHloProto(program_id, MakeHloProto("jit_fn", program_id));

  EXPECT_EQ(plane.event_metadata().size(), 1);
}

TEST(MetadataUtilsTest, ScopedOuterBuilderDoesNotCorruptProgramIdKeys) {
  tensorflow::profiler::XPlane plane;

  {
    tsl::profiler::XPlaneBuilder xp(&plane);
    auto* stat_meta = xp.GetOrCreateStatMetadata("jax_version");
    xp.AddStatValue(*stat_meta, std::string("0.9.2"));
  }

  const uint64_t program_id = 664863;
  {
    MetadataXPlaneBuilder metadata_plane(&plane);
    metadata_plane.AddHloProto(program_id, MakeHloProto("jit_fn", program_id));
  }

  ASSERT_EQ(plane.event_metadata().size(), 1);
  EXPECT_TRUE(plane.event_metadata().contains(program_id))
      << "Scoped outer XPlaneBuilder must not change event_metadata map keys.";
}

}  // namespace
}  // namespace profiler
}  // namespace xla
