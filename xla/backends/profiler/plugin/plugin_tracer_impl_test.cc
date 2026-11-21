/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/profiler/plugin/plugin_tracer_impl.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/backends/profiler/plugin/plugin_tracer.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/backends/profiler/plugin/profiler_error.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"

namespace xla {
namespace profiler {
namespace {

using tensorflow::ProfileOptions;
using tsl::profiler::ConsumeResult;
using tsl::profiler::ProfilerInterface;
using tsl::profiler::XPlaneBuilder;

constexpr size_t kTestChunkEstimatedBytes = 128 * 1024;

std::weak_ptr<std::string>& LastConsumedPayloadWeakRef() {
  static auto* weak_ref = new std::weak_ptr<std::string>();
  return *weak_ref;
}

PLUGIN_Profiler_Api MakePluginProfilerApi() {
  PLUGIN_Profiler_Api api{};
  api.create = &PLUGIN_Profiler_Create;
  api.start = &PLUGIN_Profiler_Start;
  api.stop = &PLUGIN_Profiler_Stop;
  api.collect_data = &PLUGIN_Profiler_CollectData;
  api.consume = &PLUGIN_Profiler_Consume;
  api.consume_result_destroy = &PLUGIN_Profiler_ConsumeResult_Destroy;
  api.serialize = &PLUGIN_Profiler_Serialize;
  api.destroy = &PLUGIN_Profiler_Destroy;
  api.error_destroy = &PLUGIN_Profiler_Error_Destroy;
  api.error_message = &PLUGIN_Profiler_Error_Message;
  api.error_get_code = &PLUGIN_Profiler_Error_GetCode;
  api.struct_size = PLUGIN_Profiler_Api_STRUCT_SIZE;
  return api;
}

class PluginTracerImpl : public ProfilerInterface {
 public:
  explicit PluginTracerImpl(const ProfileOptions& options)
      : options_(options) {}

  // Starts profiling.
  absl::Status Start() override {
    LOG(INFO) << "Starting Tracer";
    return absl::OkStatus();
  }

  // Stops profiling.
  absl::Status Stop() override {
    LOG(INFO) << "Stopping Tracer";
    return absl::OkStatus();
  }

  // Saves collected profile data into XSpace.
  absl::Status CollectData(tensorflow::profiler::XSpace* space) override {
    LOG(INFO) << "Collecting data";
    PopulatePlane(options_.SerializeAsString(), space);
    return absl::OkStatus();
  }

  absl::StatusOr<ConsumeResult> Consume() override {
    auto payload = std::make_shared<std::string>(options_.SerializeAsString());
    LastConsumedPayloadWeakRef() = payload;
    ConsumeResult result;
    result.data = std::move(payload);
    result.estimated_size_bytes = kTestChunkEstimatedBytes;
    return result;
  }

  absl::Status Serialize(std::any data,
                         tensorflow::profiler::XSpace* space) override {
    auto* payload = std::any_cast<std::shared_ptr<std::string>>(&data);
    if (payload == nullptr || *payload == nullptr) {
      return absl::InvalidArgumentError("Missing chunk payload");
    }
    PopulatePlane(**payload, space);
    payload->reset();
    return absl::OkStatus();
  }

 private:
  static void PopulatePlane(const std::string& stat_value,
                            tensorflow::profiler::XSpace* space) {
    tensorflow::profiler::XPlane* plane = space->add_planes();
    XPlaneBuilder builder(plane);
    builder.SetName("GpuBackendTracer");
    tensorflow::profiler::XStatMetadata* metadata =
        builder.GetOrCreateStatMetadata(int64_t{0});
    metadata->set_name("ProfileOptions");
    builder.AddStatValue(*metadata, stat_value);
  }

  ProfileOptions options_;
};

std::unique_ptr<ProfilerInterface> CreatePluginTracer(
    const ProfileOptions& options) {
  return std::make_unique<PluginTracerImpl>(options);
}

static auto register_test_tracer = [] {
  RegisterProfilerFactory(&CreatePluginTracer);
  return 0;
}();

TEST(PluginTracerTest, TestPluginWithPluginTracer) {
  PLUGIN_Profiler_Api api = MakePluginProfilerApi();

  // The options would be emitted by the tracer in the XPlane Stat which can be
  // checked to see if they are the same. This ensures that the tracer is
  // correctly passing the options to the PluginTracer implementations.
  ProfileOptions options;
  options.set_repository_path("TestRepositoryPath");
  options.set_device_tracer_level(2);

  PluginTracer tracer(&api, options);

  tensorflow::profiler::XSpace xspace;
  ASSERT_OK(tracer.Start());
  ASSERT_OK(tracer.Stop());

  ASSERT_OK(tracer.CollectData(&xspace));

  ASSERT_THAT(xspace.planes(), testing::SizeIs(1));

  const tensorflow::profiler::XPlane* plane =
      tsl::profiler::FindPlaneWithName(xspace, "GpuBackendTracer");
  ASSERT_NE(plane, nullptr);
  ASSERT_THAT(plane->stats(), testing::SizeIs(1));

  tsl::profiler::XPlaneVisitor visitor(plane);
  std::optional<tsl::profiler::XStatVisitor> stat =
      visitor.GetStat(0, *visitor.GetStatMetadata(0));

  ASSERT_TRUE(stat.has_value());
  EXPECT_EQ(stat->Name(), "ProfileOptions");
  EXPECT_EQ(stat->StrOrRefValue(), options.SerializeAsString());
}

TEST(PluginTracerTest, ConsumePropagatesEstimatedSizeAndReleasesHandle) {
  PLUGIN_Profiler_Api api = MakePluginProfilerApi();

  ProfileOptions options;
  options.set_repository_path("ContinuousRepositoryPath");
  options.set_device_tracer_level(2);

  PluginTracer tracer(&api, options);
  ASSERT_OK(tracer.Start());

  ASSERT_OK_AND_ASSIGN(ConsumeResult consumed, tracer.Consume());
  EXPECT_EQ(consumed.estimated_size_bytes, kTestChunkEstimatedBytes);
  EXPECT_FALSE(LastConsumedPayloadWeakRef().expired());

  auto* wrapper_ptr =
      std::any_cast<std::shared_ptr<PLUGIN_Profiler_ConsumeResult>>(
          &consumed.data);
  ASSERT_NE(wrapper_ptr, nullptr);
  std::weak_ptr<PLUGIN_Profiler_ConsumeResult> wrapper_weak = *wrapper_ptr;
  EXPECT_FALSE(wrapper_weak.expired());

  ASSERT_OK(tracer.Stop());

  tensorflow::profiler::XSpace xspace;
  ASSERT_OK(tracer.Serialize(std::move(consumed.data), &xspace));

  // Both the inner raw chunk payload and the C-API serialized buffer wrapper
  // must be freed immediately upon Serialize completing.
  EXPECT_TRUE(LastConsumedPayloadWeakRef().expired());
  EXPECT_TRUE(wrapper_weak.expired());

  const tensorflow::profiler::XPlane* plane =
      tsl::profiler::FindPlaneWithName(xspace, "GpuBackendTracer");
  ASSERT_NE(plane, nullptr);
  tsl::profiler::XPlaneVisitor visitor(plane);
  std::optional<tsl::profiler::XStatVisitor> stat =
      visitor.GetStat(0, *visitor.GetStatMetadata(0));
  ASSERT_TRUE(stat.has_value());
  EXPECT_EQ(stat->StrOrRefValue(), options.SerializeAsString());
}

TEST(PluginTracerTest, SerializeReleasesRawChunkBeforeWrapperDestroy) {
  ProfileOptions options;
  options.set_repository_path("ChunkReleaseTest");
  std::string options_bytes = options.SerializeAsString();

  PLUGIN_Profiler_Create_Args create_args{};
  create_args.struct_size = PLUGIN_Profiler_Create_Args_STRUCT_SIZE;
  create_args.options = options_bytes.data();
  create_args.options_size = options_bytes.size();
  ASSERT_EQ(PLUGIN_Profiler_Create(&create_args), nullptr);
  ASSERT_NE(create_args.profiler, nullptr);

  PLUGIN_Profiler_Start_Args start_args{};
  start_args.struct_size = PLUGIN_Profiler_Start_Args_STRUCT_SIZE;
  start_args.profiler = create_args.profiler;
  ASSERT_EQ(PLUGIN_Profiler_Start(&start_args), nullptr);

  PLUGIN_Profiler_Consume_Args consume_args{};
  consume_args.struct_size = PLUGIN_Profiler_Consume_Args_STRUCT_SIZE;
  consume_args.profiler = create_args.profiler;
  ASSERT_EQ(PLUGIN_Profiler_Consume(&consume_args), nullptr);
  ASSERT_NE(consume_args.result, nullptr);
  EXPECT_EQ(consume_args.estimated_size_bytes, kTestChunkEstimatedBytes);
  EXPECT_TRUE(consume_args.result->consume_result.data.has_value());
  EXPECT_FALSE(LastConsumedPayloadWeakRef().expired());

  PLUGIN_Profiler_Serialize_Args serialize_args{};
  serialize_args.struct_size = PLUGIN_Profiler_Serialize_Args_STRUCT_SIZE;
  serialize_args.profiler = create_args.profiler;
  serialize_args.consume_result = consume_args.result;
  ASSERT_EQ(PLUGIN_Profiler_Serialize(&serialize_args), nullptr);

  // Raw consumed chunk data must be released immediately during
  // PLUGIN_Profiler_Serialize rather than pinned alongside the serialized byte
  // buffer until PLUGIN_Profiler_ConsumeResult_Destroy.
  EXPECT_TRUE(LastConsumedPayloadWeakRef().expired());
  EXPECT_FALSE(consume_args.result->consume_result.data.has_value());
  EXPECT_EQ(consume_args.result->consume_result.estimated_size_bytes, 0);
  EXPECT_NE(serialize_args.serialized_bytes, nullptr);
  EXPECT_GT(serialize_args.serialized_size, 0);

  tensorflow::profiler::XSpace parsed_space;
  EXPECT_TRUE(parsed_space.ParseFromArray(serialize_args.serialized_bytes,
                                          serialize_args.serialized_size));
  EXPECT_NE(tsl::profiler::FindPlaneWithName(parsed_space, "GpuBackendTracer"),
            nullptr);

  PLUGIN_Profiler_ConsumeResult_Destroy_Args destroy_res_args{};
  destroy_res_args.struct_size =
      PLUGIN_Profiler_ConsumeResult_Destroy_Args_STRUCT_SIZE;
  destroy_res_args.consume_result = consume_args.result;
  PLUGIN_Profiler_ConsumeResult_Destroy(&destroy_res_args);

  PLUGIN_Profiler_Stop_Args stop_args{};
  stop_args.struct_size = PLUGIN_Profiler_Stop_Args_STRUCT_SIZE;
  stop_args.profiler = create_args.profiler;
  ASSERT_EQ(PLUGIN_Profiler_Stop(&stop_args), nullptr);

  PLUGIN_Profiler_Destroy_Args destroy_args{};
  destroy_args.struct_size = PLUGIN_Profiler_Destroy_Args_STRUCT_SIZE;
  destroy_args.profiler = create_args.profiler;
  ASSERT_EQ(PLUGIN_Profiler_Destroy(&destroy_args), nullptr);
}

TEST(PluginTracerTest, ConsumeHandlesLegacyStructSizesSafely) {
  ProfileOptions options;
  options.set_repository_path("LegacyAbiTest");
  std::string options_bytes = options.SerializeAsString();

  // 1. Legacy caller (struct_size ending at `result`) calling newer plugin:
  //    must not write `estimated_size_bytes` past caller's struct boundary.
  PLUGIN_Profiler_Create_Args create_args{};
  create_args.struct_size = PLUGIN_Profiler_Create_Args_STRUCT_SIZE;
  create_args.options = options_bytes.data();
  create_args.options_size = options_bytes.size();
  ASSERT_EQ(PLUGIN_Profiler_Create(&create_args), nullptr);

  constexpr size_t kSentinel = 0x5A5A5A5Au;
  PLUGIN_Profiler_Consume_Args legacy_caller_args{};
  legacy_caller_args.struct_size =
      PROFILER_STRUCT_SIZE(PLUGIN_Profiler_Consume_Args, result);
  legacy_caller_args.profiler = create_args.profiler;
  legacy_caller_args.result = nullptr;
  legacy_caller_args.estimated_size_bytes = kSentinel;

  ASSERT_EQ(PLUGIN_Profiler_Consume(&legacy_caller_args), nullptr);
  ASSERT_NE(legacy_caller_args.result, nullptr);
  EXPECT_EQ(legacy_caller_args.estimated_size_bytes, kSentinel);

  PLUGIN_Profiler_ConsumeResult_Destroy_Args destroy_res_args{};
  destroy_res_args.struct_size =
      PLUGIN_Profiler_ConsumeResult_Destroy_Args_STRUCT_SIZE;
  destroy_res_args.consume_result = legacy_caller_args.result;
  PLUGIN_Profiler_ConsumeResult_Destroy(&destroy_res_args);

  PLUGIN_Profiler_Destroy_Args destroy_args{};
  destroy_args.struct_size = PLUGIN_Profiler_Destroy_Args_STRUCT_SIZE;
  destroy_args.profiler = create_args.profiler;
  ASSERT_EQ(PLUGIN_Profiler_Destroy(&destroy_args), nullptr);

  // 2. Newer host `PluginTracer` calling legacy plugin that ignores
  //    `estimated_size_bytes`: must default cleanly to 0 bytes.
  PLUGIN_Profiler_Api legacy_plugin_api = MakePluginProfilerApi();
  legacy_plugin_api.consume =
      +[](PLUGIN_Profiler_Consume_Args* args) -> PLUGIN_Profiler_Error* {
    PLUGIN_Profiler_Consume_Args inner_args{};
    inner_args.struct_size =
        PROFILER_STRUCT_SIZE(PLUGIN_Profiler_Consume_Args, result);
    inner_args.profiler = args->profiler;
    inner_args.result = nullptr;
    PLUGIN_Profiler_Error* error = PLUGIN_Profiler_Consume(&inner_args);
    args->result = inner_args.result;
    // Intentionally leave args->estimated_size_bytes untouched.
    return error;
  };

  PluginTracer tracer(&legacy_plugin_api, options);
  ASSERT_OK(tracer.Start());
  ASSERT_OK_AND_ASSIGN(ConsumeResult consumed, tracer.Consume());
  EXPECT_EQ(consumed.estimated_size_bytes, 0);
  ASSERT_OK(tracer.Stop());
}

}  // namespace
}  // namespace profiler
}  // namespace xla
