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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/memory/memory.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "riegeli/bytes/string_reader.h"
#include "tsl/platform/casts.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/gpu/se_gpu_executable_metadata_provider.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"
#include "xla/pjrt/infer_dispatch_info.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/se/stream_executor_executable.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_aot_compilation_result.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;

constexpr absl::string_view kProgram = R"(HloModule Computation

ENTRY Computation() -> s32[] {
  ROOT result = s32[] constant(2)
})";

constexpr absl::string_view mlir_str = R"mlir(
  module {
    func.func @main() -> tensor<i32> {
      %0 = mhlo.constant dense<2> : tensor<i32>
      return %0 : tensor<i32>
    }
  })mlir";

absl::StatusOr<xla::XlaComputation> GetXlaComputation(
    absl::string_view program) {
  ABSL_ASSIGN_OR_RETURN(auto hlo_module,
                        xla::ParseAndReturnUnverifiedModule(program, {}));

  return XlaComputation(hlo_module->ToProto());
}

void ValidateResult(
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>& result) {
  ASSERT_EQ(result.size(), 1);
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  ASSERT_EQ(result_buffers.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> result_literal,
                          result_buffers[0]->ToLiteral().Await());
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0(2), *result_literal));
}

xla::LocalClient* GetLocalClient(PjRtClient* client) {
  return absl::down_cast<PjRtStreamExecutorRawClient*>(
             absl::down_cast<CommonPjRtClient*>(client)->raw_client())
      ->client();
}

TEST(StreamExecutorGpuCompilerTest, SuccessAotCompileMlirAndLoad) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  Compiler::GpuTargetConfig gpu_target_config = xla::Compiler::GpuTargetConfig(
      GetLocalClient(client.get())->backend().default_stream_executor());
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());

  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect>();
  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_str, context.get());
  TF_ASSERT_OK_AND_ASSIGN(auto topology, client->GetTopologyDescription());
  xla::CompileOptions opts;
  opts.gpu_target_config = gpu_target_config;

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler.Compile(
          opts,
          MaybeOwningMlirModule(std::move(context), std::move(mlir_module)),
          *topology, /*client=*/nullptr));
  EXPECT_THAT(executable->GetHloModules(), IsOkAndHolds(SizeIs(1)));
  TF_ASSERT_OK_AND_ASSIGN(auto loaded_executable,
                          client->Load(std::move(executable), LoadOptions()));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result,
      loaded_executable->Execute(/*argument_handles=*/{{}}, {}));
  ValidateResult(result);
}

TEST(StreamExecutorGpuCompilerTest, AotCompileDeserializeRoundTrip) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  Compiler::GpuTargetConfig gpu_target_config = xla::Compiler::GpuTargetConfig(
      GetLocalClient(client.get())->backend().default_stream_executor());
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());

  auto context = std::make_unique<mlir::MLIRContext>();
  context->loadDialect<mlir::mhlo::MhloDialect, mlir::func::FuncDialect>();
  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_str, context.get());
  TF_ASSERT_OK_AND_ASSIGN(auto topology, client->GetTopologyDescription());
  xla::CompileOptions opts;
  opts.gpu_target_config = gpu_target_config;

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler.Compile(
          opts,
          MaybeOwningMlirModule(std::move(context), std::move(mlir_module)),
          *topology, /*client=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          executable->SerializeExecutable());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StreamExecutorExecutable> deserialized_executable,
      StreamExecutorExecutable::Deserialize(riegeli::StringReader<>(serialized),
                                            *topology, std::nullopt));

  TF_ASSERT_OK_AND_ASSIGN(std::string reserialized,
                          deserialized_executable->SerializeExecutable());
  EXPECT_EQ(serialized, reserialized);
}

TEST(StreamExecutorGpuCompilerTest, SuccessAotCompileXlaAndLoad) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  Compiler::GpuTargetConfig gpu_target_config{
      GetLocalClient(client.get())->backend().default_stream_executor()};
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                          GetXlaComputation(kProgram));
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  xla::CompileOptions opts;
  opts.gpu_target_config = gpu_target_config;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      compiler.Compile(opts, computation, *topology, /*client=*/nullptr));
  EXPECT_THAT(executable->GetHloModules(), IsOkAndHolds(SizeIs(1)));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> loaded_executable,
      client->Load(std::move(executable), LoadOptions()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result,
      loaded_executable->Execute(/*argument_handles=*/{{}}, {}));
  ValidateResult(result);
}

TEST(StreamExecutorGpuCompilerTest, SuccessLoadFromSerializedExecutable) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());
  xla::CompileOptions opts;
  opts.gpu_target_config = Compiler::GpuTargetConfig(
      GetLocalClient(client.get())->backend().default_stream_executor());

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                          GetXlaComputation(kProgram));
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      compiler.Compile(opts, computation, *topology, /*client=*/nullptr));
  EXPECT_THAT(executable->GetHloModules(), IsOkAndHolds(SizeIs(1)));

  // Serialize the executable and load it.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_executable,
                          executable->SerializeExecutable());
  TF_ASSERT_OK_AND_ASSIGN(
      auto loaded_executable,
      client->LoadSerializedExecutable(serialized_executable, std::nullopt,
                                       LoadOptions()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, loaded_executable->Execute(/*argument_handles=*/{{}}, {}));
  ValidateResult(result);
}

constexpr absl::string_view kProgramIdentity = R"(HloModule Identity

ENTRY main {
  ROOT Arg_0.1 = s32[1]{0} parameter(0)
})";

TEST(StreamExecutorGpuCompilerTest, SuccessSerializeDeserialize) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());
  xla::CompileOptions opts;
  opts.gpu_target_config = Compiler::GpuTargetConfig(
      GetLocalClient(client.get())->backend().default_stream_executor());

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                          GetXlaComputation(kProgramIdentity));
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      compiler.Compile(opts, computation, *topology, /*client=*/nullptr));
  EXPECT_THAT(executable->GetHloModules(), IsOkAndHolds(SizeIs(1)));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> loaded_executable,
      client->Load(std::move(executable), LoadOptions()));

  // Serialize the executable and deserialize it without failure.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_executable,
                          loaded_executable->SerializeExecutable());
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized_executable,
      client->LoadSerializedExecutable(serialized_executable, std::nullopt,
                                       LoadOptions()));

  EXPECT_EQ(deserialized_executable->GetExecutable()->name(), "Identity");
}

constexpr char const* kD2HProgramTupleOutput = R"(
  HloModule f

  ENTRY main.5 {
    p = s32[4]{0} parameter(0)
    cc = s32[4] custom-call(p),
        custom_call_target="annotate_device_placement",
        frontend_attributes={_xla_buffer_placement="pinned_host"}
    ROOT tuple = (s32[4]{0}, s32[4]{0}) tuple(s32[4]{0} p, s32[4]{0} cc)
  }
)";
TEST(StreamExecutorGpuCompilerTest, UnloadedExecutableMemoryStats) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());
  xla::CompileOptions options;
  options.gpu_target_config = Compiler::GpuTargetConfig(
      GetLocalClient(client.get())->backend().default_stream_executor());

  // Build the output shape with the correct memory space set.
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  Shape host_shape = shape;
  host_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
  Shape out_shape = ShapeUtil::MakeTupleShape({shape, host_shape});
  options.executable_build_options.set_result_layout(out_shape);

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                          GetXlaComputation(kD2HProgramTupleOutput));
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      compiler.Compile(options, computation, *topology, /*client=*/nullptr));
  EXPECT_THAT(executable->GetHloModules(), IsOkAndHolds(SizeIs(1)));

  TF_ASSERT_OK_AND_ASSIGN(CompiledMemoryStats compiled_memory_stats,
                          executable->GetCompiledMemoryStats());

  EXPECT_EQ(compiled_memory_stats.argument_size_in_bytes, 16);
  EXPECT_EQ(compiled_memory_stats.output_size_in_bytes, 32);
  EXPECT_GT(compiled_memory_stats.temp_size_in_bytes, 0);
  EXPECT_EQ(compiled_memory_stats.host_temp_size_in_bytes, 0);
  EXPECT_EQ(compiled_memory_stats.host_output_size_in_bytes, 16);
  EXPECT_GT(compiled_memory_stats.peak_memory_in_bytes, 0);
  EXPECT_GT(compiled_memory_stats.total_allocation_bytes, 0);
  EXPECT_GT(compiled_memory_stats.indefinite_allocations, 0);
}

TEST(StreamExecutorGpuCompilerTest, AutoLayoutIsSupported) {
  const char* hlo_text = R"(
  HloModule DotLayout,
    entry_computation_layout={(f32[2,3,5],f32[3,4,5])->f32[5,2,4]{2,1,0}}

  ENTRY dot {
    p0 = f32[2,3,5]{2,1,0} parameter(0)
    p1 = f32[3,4,5]{2,1,0} parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4]{2,1,0} dot(p0, p1),
      lhs_batch_dims={2}, lhs_contracting_dims={1},
      rhs_batch_dims={2}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo_text, {}, HloParserOptions().set_fill_missing_layouts(false)));

  CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_pjrt_allow_auto_layout_in_hlo(true);
  XlaComputation computation = m->ToProto();
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable, compiler.Compile(compile_options, computation, *topology,
                                        /*client=*/nullptr));
  TF_ASSERT_OK_AND_ASSIGN(auto layouts, executable->GetParameterLayouts());
  // Check that the assigned layouts are not default.
  EXPECT_NE(layouts[0]->ToString(), "{2,1,0}");
  EXPECT_NE(layouts[1]->ToString(), "{2,1,0}");
}

// The metadata provider reads the name, replica and partition counts and the
// fingerprint of a deserialized executable from the serialized proto. They have
// to be the same as the values derived from the deserialized HLO module.
TEST(StreamExecutorGpuCompilerTest, DeserializedMetadataMatchesModule) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());

  for (absl::string_view program : {kProgram, kProgramIdentity}) {
    for (int num_replicas : {1, 2}) {
      SCOPED_TRACE(absl::StrCat(program, " num_replicas=", num_replicas));
      xla::CompileOptions opts;
      opts.gpu_target_config = Compiler::GpuTargetConfig(
          GetLocalClient(client.get())->backend().default_stream_executor());
      opts.executable_build_options.set_num_replicas(num_replicas);
      TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                              GetXlaComputation(program));
      TF_ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<PjRtExecutable> executable,
          compiler.Compile(opts, computation, *topology, /*client=*/nullptr));
      TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                              executable->SerializeExecutable());

      // The provider reads the metadata from the proto.
      TF_ASSERT_OK_AND_ASSIGN(ExecutableAndOptionsProto executable_and_options,
                              SerializedGpuExecutableFromString(serialized));
      gpu::GpuExecutableProto gpu_executable_proto;
      ASSERT_OK(
          ReadSplitProto(std::make_unique<riegeli::StringReader<>>(
                             executable_and_options.serialized_executable()),
                         gpu_executable_proto));
      TF_ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<gpu::GpuAotCompilationResult> aot_result,
          gpu::GpuAotCompilationResult::FromProto(gpu_executable_proto));
      std::optional<
          StreamExecutorExecutableMetadataProvider::CompiledModuleMetadata>
          metadata =
              GpuExecutableMetadataProvider::Get()->GetCompiledModuleMetadata(
                  *aot_result);
      ASSERT_TRUE(metadata.has_value());
      const HloModule* module = aot_result->optimized_module();
      ASSERT_NE(module, nullptr);
      EXPECT_EQ(metadata->name, module->name());
      EXPECT_EQ(metadata->num_replicas, module->config().replica_count());
      EXPECT_EQ(metadata->num_partitions, module->config().num_partitions());
      EXPECT_EQ(metadata->fingerprint, module->GetFingerprint128());
      EXPECT_EQ(metadata->num_replicas, num_replicas);

      // Deserializing with and without the provider gives the same executable
      // metadata.
      TF_ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<StreamExecutorExecutable> with_provider,
          StreamExecutorExecutable::Deserialize(
              riegeli::StringReader<>(serialized), *topology, std::nullopt,
              GpuExecutableMetadataProvider::Get()));
      TF_ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<StreamExecutorExecutable> without_provider,
          StreamExecutorExecutable::Deserialize(
              riegeli::StringReader<>(serialized), *topology, std::nullopt));
      EXPECT_EQ(with_provider->name(), without_provider->name());
      EXPECT_EQ(with_provider->num_replicas(),
                without_provider->num_replicas());
      EXPECT_EQ(with_provider->num_partitions(),
                without_provider->num_partitions());
      TF_ASSERT_OK_AND_ASSIGN(std::string fingerprint,
                              without_provider->FingerprintExecutable());
      EXPECT_THAT(with_provider->FingerprintExecutable(),
                  IsOkAndHolds(fingerprint));
      EXPECT_THAT(with_provider->GetHloModules(), IsOkAndHolds(SizeIs(1)));
    }
  }
}

// After loading, the module of a deserialized executable comes from the loaded
// `GpuExecutable`, and it is the module of the compiled module.
TEST(StreamExecutorGpuCompilerTest,
     LoadedExecutableKeepsTheDeserializedModule) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());
  xla::CompileOptions opts;
  opts.gpu_target_config = Compiler::GpuTargetConfig(
      GetLocalClient(client.get())->backend().default_stream_executor());
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                          GetXlaComputation(kProgram));
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      compiler.Compile(opts, computation, *topology, /*client=*/nullptr));
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          executable->SerializeExecutable());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> deserialized,
      compiler.DeserializeExecutable(
          *topology, riegeli::StringReader<>(serialized), std::nullopt, {}));
  auto* se_executable =
      absl::down_cast<StreamExecutorExecutable*>(deserialized.get());
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module_before_load,
                          se_executable->GetHloModule());

  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<LocalExecutable> local_executable,
      se_executable->GetOrLoadExecutable(GetLocalClient(client.get())));
  EXPECT_THAT(se_executable->GetHloModule(), IsOkAndHolds(module_before_load));
  EXPECT_EQ(local_executable->executable()->shared_module(),
            module_before_load);
}

// Forwards to `GpuExecutableMetadataProvider` and counts how often the HLO
// module is requested.
class CountingMetadataProvider final
    : public StreamExecutorExecutableMetadataProvider {
 public:
  absl::StatusOr<std::shared_ptr<HloModule>> GetHloModule(
      const Executable& executable, absl::string_view reason) const override {
    ++num_hlo_module_requests_;
    return GpuExecutableMetadataProvider::Get()->GetHloModule(executable,
                                                              reason);
  }

  std::optional<CompiledModuleMetadata> GetCompiledModuleMetadata(
      const CompiledModule& compiled_module) const override {
    return GpuExecutableMetadataProvider::Get()->GetCompiledModuleMetadata(
        compiled_module);
  }

  std::optional<SpmdShardings> GetSpmdShardings(
      const Executable& executable) const override {
    return GpuExecutableMetadataProvider::Get()->GetSpmdShardings(executable);
  }

  int num_hlo_module_requests() const { return num_hlo_module_requests_; }

 private:
  mutable std::atomic<int> num_hlo_module_requests_ = 0;
};

constexpr absl::string_view kProgramWithAliasing = R"(
  HloModule Aliasing, input_output_alias={ {0}: (0, {}, may-alias) }

  ENTRY main {
    p0 = f32[4] parameter(0)
    p1 = f32[2,3] parameter(1)
    add = f32[4] add(p0, p0)
    ROOT tuple = (f32[4], f32[2,3]) tuple(add, p1)
  })";

constexpr absl::string_view kSpmdProgram = R"(
  HloModule Spmd

  ENTRY main {
    p0 = f32[8,4] parameter(0), sharding={devices=[2,1]<=[2]}
    p1 = f32[8,4] parameter(1), sharding={replicated}
    add = f32[8,4] add(p0, p1), sharding={devices=[2,1]<=[2]}
    ROOT tuple = (f32[8,4], f32[8,4]) tuple(add, p1),
        sharding={{devices=[2,1]<=[2]}, {replicated}}
  })";

std::vector<std::string> LayoutStrings(
    const std::vector<std::shared_ptr<const PjRtLayout>>& layouts) {
  std::vector<std::string> result;
  result.reserve(layouts.size());
  for (const std::shared_ptr<const PjRtLayout>& layout : layouts) {
    result.push_back(layout->ToString());
  }
  return result;
}

void ExpectSameShardings(
    const std::optional<std::vector<OpSharding>>& actual,
    const std::optional<std::vector<OpSharding>>& expected) {
  ASSERT_EQ(actual.has_value(), expected.has_value());
  if (!expected.has_value()) {
    return;
  }
  ASSERT_EQ(actual->size(), expected->size());
  for (int i = 0; i < expected->size(); ++i) {
    EXPECT_THAT((*actual)[i], EqualsProto((*expected)[i]));
  }
}

struct MetadataTestCase {
  std::string name;
  absl::string_view program;
  int num_partitions = 1;
  std::optional<Shape> result_layout;
};

// Once loaded, an executable with the metadata provider reads its layouts,
// shardings, memory kinds and output shapes from the `GpuExecutable` instead of
// the HLO module. They have to be the same as the values that an executable
// without a provider derives from the module.
TEST(StreamExecutorGpuCompilerTest, LoadedMetadataMatchesModule) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());

  Shape device_shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  Shape host_shape = device_shape;
  host_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
  const std::vector<MetadataTestCase> test_cases = {
      {"constant", kProgram},
      {"identity", kProgramIdentity},
      {"aliasing", kProgramWithAliasing},
      {"spmd", kSpmdProgram, /*num_partitions=*/2},
      {"host_output", kD2HProgramTupleOutput, /*num_partitions=*/1,
       ShapeUtil::MakeTupleShape({device_shape, host_shape})},
  };

  for (const MetadataTestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    xla::CompileOptions opts;
    opts.gpu_target_config = Compiler::GpuTargetConfig(
        GetLocalClient(client.get())->backend().default_stream_executor());
    if (test_case.num_partitions > 1) {
      opts.executable_build_options.set_num_partitions(
          test_case.num_partitions);
      opts.executable_build_options.set_use_spmd_partitioning(true);
    }
    if (test_case.result_layout.has_value()) {
      opts.executable_build_options.set_result_layout(*test_case.result_layout);
    }
    TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                            GetXlaComputation(test_case.program));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtExecutable> executable,
        compiler.Compile(opts, computation, *topology, /*client=*/nullptr));
    TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                            executable->SerializeExecutable());

    auto provider = std::make_shared<CountingMetadataProvider>();
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<StreamExecutorExecutable> with_provider,
        StreamExecutorExecutable::Deserialize(
            riegeli::StringReader<>(serialized), *topology, std::nullopt,
            provider));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<StreamExecutorExecutable> without_provider,
        StreamExecutorExecutable::Deserialize(
            riegeli::StringReader<>(serialized), *topology, std::nullopt));
    ASSERT_OK(with_provider->GetOrLoadExecutable(GetLocalClient(client.get())));
    ASSERT_OK(
        without_provider->GetOrLoadExecutable(GetLocalClient(client.get())));

    TF_ASSERT_OK_AND_ASSIGN(std::vector<Shape> output_shapes,
                            without_provider->GetOutputShapes());
    EXPECT_THAT(with_provider->GetOutputShapes(), IsOkAndHolds(output_shapes));

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::shared_ptr<const PjRtLayout>> parameter_layouts,
        without_provider->GetParameterLayouts());
    TF_ASSERT_OK_AND_ASSIGN(std::vector<std::shared_ptr<const PjRtLayout>>
                                parameter_layouts_from_provider,
                            with_provider->GetParameterLayouts());
    EXPECT_EQ(LayoutStrings(parameter_layouts_from_provider),
              LayoutStrings(parameter_layouts));

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::shared_ptr<const PjRtLayout>> output_layouts,
        without_provider->GetOutputLayouts());
    TF_ASSERT_OK_AND_ASSIGN(std::vector<std::shared_ptr<const PjRtLayout>>
                                output_layouts_from_provider,
                            with_provider->GetOutputLayouts());
    EXPECT_EQ(LayoutStrings(output_layouts_from_provider),
              LayoutStrings(output_layouts));

    ExpectSameShardings(with_provider->GetParameterShardings(),
                        without_provider->GetParameterShardings());
    ExpectSameShardings(with_provider->GetOutputShardings(),
                        without_provider->GetOutputShardings());
    if (test_case.num_partitions > 1) {
      // Make sure that we don't only compare missing shardings.
      EXPECT_TRUE(without_provider->GetParameterShardings().has_value());
      EXPECT_TRUE(without_provider->GetOutputShardings().has_value());
    }

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::vector<absl::string_view>> parameter_memory_kinds,
        without_provider->GetParameterMemoryKinds());
    EXPECT_THAT(with_provider->GetParameterMemoryKinds(),
                IsOkAndHolds(parameter_memory_kinds));
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::vector<absl::string_view>> output_memory_kinds,
        without_provider->GetOutputMemoryKinds());
    EXPECT_THAT(with_provider->GetOutputMemoryKinds(),
                IsOkAndHolds(output_memory_kinds));

    // None of the above needed the HLO module.
    EXPECT_EQ(provider->num_hlo_module_requests(), 0);
  }
}

// When the PJRT client loads an executable with the metadata provider, it takes
// the shapes and the aliasing from the `GpuExecutable` instead of the HLO
// module. The resulting dispatch info has to be the same as the one of an
// executable without a provider.
TEST(StreamExecutorGpuCompilerTest, LoadedDispatchInfoMatchesModule) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  StreamExecutorGpuCompiler compiler(
      client->platform_id(), GetLocalClient(client.get())->platform()->id());
  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());

  Shape device_shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  Shape host_shape = device_shape;
  host_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
  const std::vector<MetadataTestCase> test_cases = {
      {"constant", kProgram},
      {"identity", kProgramIdentity},
      {"aliasing", kProgramWithAliasing},
      {"host_output", kD2HProgramTupleOutput, /*num_partitions=*/1,
       ShapeUtil::MakeTupleShape({device_shape, host_shape})},
  };

  for (const MetadataTestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    xla::CompileOptions opts;
    opts.gpu_target_config = Compiler::GpuTargetConfig(
        GetLocalClient(client.get())->backend().default_stream_executor());
    if (test_case.result_layout.has_value()) {
      opts.executable_build_options.set_result_layout(*test_case.result_layout);
    }
    TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                            GetXlaComputation(test_case.program));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtExecutable> executable,
        compiler.Compile(opts, computation, *topology, /*client=*/nullptr));
    TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                            executable->SerializeExecutable());

    auto provider = std::make_shared<CountingMetadataProvider>();
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<StreamExecutorExecutable> with_provider,
        StreamExecutorExecutable::Deserialize(
            riegeli::StringReader<>(serialized), *topology, std::nullopt,
            provider));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<StreamExecutorExecutable> without_provider,
        StreamExecutorExecutable::Deserialize(
            riegeli::StringReader<>(serialized), *topology, std::nullopt));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtLoadedExecutable> loaded_with_provider,
        client->Load(std::move(with_provider), LoadOptions()));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtLoadedExecutable> loaded_without_provider,
        client->Load(std::move(without_provider), LoadOptions()));

    CommonPjRtLoadedExecutable::DispatchInfo actual =
        absl::down_cast<CommonPjRtLoadedExecutable*>(loaded_with_provider.get())
            ->GetDispatchInfo();
    CommonPjRtLoadedExecutable::DispatchInfo expected =
        absl::down_cast<CommonPjRtLoadedExecutable*>(
            loaded_without_provider.get())
            ->GetDispatchInfo();
    EXPECT_EQ(actual.parameter_device_shapes, expected.parameter_device_shapes);
    EXPECT_EQ(*actual.output_device_shape, *expected.output_device_shape);
    EXPECT_EQ(actual.parameter_memory_space_kind_ids,
              expected.parameter_memory_space_kind_ids);
    EXPECT_EQ(actual.output_memory_space_kind_ids,
              expected.output_memory_space_kind_ids);
    EXPECT_EQ(actual.parameters_that_may_be_donated,
              expected.parameters_that_may_be_donated);
    EXPECT_EQ(actual.input_buffer_sizes_in_bytes,
              expected.input_buffer_sizes_in_bytes);
    EXPECT_EQ(actual.input_output_alias_config.ToString(),
              expected.input_output_alias_config.ToString());
    if (test_case.name == "aliasing") {
      // Make sure that we don't only compare empty alias configs.
      EXPECT_THAT(expected.parameters_that_may_be_donated, ElementsAre(0));
    }

    if (test_case.name == "constant") {
      TF_ASSERT_OK_AND_ASSIGN(
          std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result,
          loaded_with_provider->Execute(/*argument_handles=*/{{}}, {}));
      ValidateResult(result);
    }

    // Neither loading nor executing needed the HLO module.
    EXPECT_EQ(provider->num_hlo_module_requests(), 0);
  }
}

}  // namespace
}  // namespace xla
