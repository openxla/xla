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

#include "xla/backends/gpu/runtime/conditional_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::Pointee;
using ::testing::Property;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;
using Kind = Thunk::Kind;

// A dummy `Thunk` that does nothing.
struct DummyThunk : public Thunk {
  explicit DummyThunk(Thunk::Kind kind, Thunk::ThunkInfo thunk_info)
      : Thunk(kind, std::move(thunk_info)) {}
  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
  static absl::StatusOr<std::unique_ptr<DummyThunk>> FromProto(
      const ThunkProto& thunk_proto, Thunk::Kind kind) {
    TF_ASSIGN_OR_RETURN(Thunk::ThunkInfo thunk_info,
                        Thunk::ThunkInfo::FromProto(thunk_proto.thunk_info()));
    return std::make_unique<DummyThunk>(kind, std::move(thunk_info));
  }

  absl::StatusOr<ThunkProto> ToProto() const override {
    ThunkProto proto;
    *proto.mutable_thunk_info() = thunk_info().ToProto();
    return proto;
  }
};

std::unique_ptr<ConditionalThunk> CreateConditionalThunk(
    const Thunk::ThunkInfo& thunk_info,
    const ShapedSlice& branch_index_buffer_index,
    std::vector<ThunkSequence> branch_thunk_sequences) {
  return std::make_unique<ConditionalThunk>(
      thunk_info, branch_index_buffer_index, std::move(branch_thunk_sequences));
}

struct BranchRecordCounts {
  int creates = 0;
  int updates = 0;
};

struct FakeSeCommand : public se::CommandBuffer::Command {};

class BranchRecordingCommand : public Command {
 public:
  explicit BranchRecordingCommand(BranchRecordCounts* counts)
      : Command(CommandType::kUnknownCmd), counts_(counts) {}

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams&, const RecordParams&, RecordAction action,
      se::CommandBuffer*) override {
    if (std::get_if<Command::RecordCreate>(&action) != nullptr) {
      ++counts_->creates;
      return &recorded_command_;
    }

    auto* update = std::get_if<Command::RecordUpdate>(&action);
    if (update == nullptr) {
      return absl::InternalError("unexpected record action");
    }
    if (update->command != &recorded_command_) {
      return absl::InternalError("unexpected recorded command");
    }
    ++counts_->updates;
    return &recorded_command_;
  }

  BufferUses buffer_uses() const override { return {}; }

 private:
  BranchRecordCounts* counts_;
  FakeSeCommand recorded_command_;
};

absl::StatusOr<CommandExecutor> MakeBranchExecutor(BranchRecordCounts* counts) {
  CommandSequence commands;
  commands.Append(std::make_unique<BranchRecordingCommand>(counts));
  return CommandExecutor::Create(
      std::move(commands), CommandExecutor::SynchronizationMode::kSerialize);
}

class FakeCommandBuffer : public se::CommandBuffer {
 public:
  explicit FakeCommandBuffer(Mode mode = Mode::kPrimary) : mode_(mode) {}

  const Command* case_command() const { return &case_command_; }
  const Command* dependency_command() const { return &dependency_command_; }
  int create_case_calls() const { return create_case_calls_; }
  int update_case_calls() const { return update_case_calls_; }
  int create_case_branch_count() const { return create_case_branch_count_; }
  int update_case_branch_count() const { return update_case_branch_count_; }
  int create_case_dependency_count() const {
    return create_case_dependency_count_;
  }

  absl::StatusOr<const Command*> CreateEmptyCmd(
      absl::Span<const Command* const>,
      se::StreamPriority = se::StreamPriority::Default) override {
    return absl::UnimplementedError("CreateEmptyCmd is not implemented");
  }

  absl::StatusOr<const Command*> CreateLaunch(
      const se::ThreadDim&, const se::BlockDim&, const se::Kernel&,
      const se::KernelArgs&, absl::Span<const Command* const>,
      se::StreamPriority = se::StreamPriority::Default) override {
    return absl::UnimplementedError("CreateLaunch is not implemented");
  }

  absl::Status UpdateLaunch(const Command*, const se::ThreadDim&,
                            const se::BlockDim&, const se::Kernel&,
                            const se::KernelArgs&) override {
    return absl::UnimplementedError("UpdateLaunch is not implemented");
  }

  absl::StatusOr<const Command*> CreateChildCommand(
      const se::CommandBuffer&, absl::Span<const Command* const>) override {
    return absl::UnimplementedError("CreateChildCommand is not implemented");
  }

  absl::Status UpdateChildCommand(const Command*,
                                  const se::CommandBuffer&) override {
    return absl::UnimplementedError("UpdateChildCommand is not implemented");
  }

  absl::StatusOr<const Command*> CreateChildCommand(
      absl::AnyInvocable<absl::Status(se::CommandBuffer*)>,
      absl::Span<const Command* const>) override {
    return absl::UnimplementedError("CreateChildCommand is not implemented");
  }

  absl::Status UpdateChildCommand(
      const Command*,
      absl::AnyInvocable<absl::Status(se::CommandBuffer*)>) override {
    return absl::UnimplementedError("UpdateChildCommand is not implemented");
  }

  absl::StatusOr<const Command*> CreateMemcpyD2D(
      se::DeviceAddressBase*, const se::DeviceAddressBase&, uint64_t,
      absl::Span<const Command* const>) override {
    return absl::UnimplementedError("CreateMemcpyD2D is not implemented");
  }

  absl::Status UpdateMemcpyD2D(const Command*, se::DeviceAddressBase*,
                               const se::DeviceAddressBase&,
                               uint64_t) override {
    return absl::UnimplementedError("UpdateMemcpyD2D is not implemented");
  }

  absl::StatusOr<const Command*> CreateMemset(
      se::DeviceAddressBase*, se::BitPattern, size_t,
      absl::Span<const Command* const>) override {
    return absl::UnimplementedError("CreateMemset is not implemented");
  }

  absl::Status UpdateMemset(const Command*, se::DeviceAddressBase*,
                            const se::BitPattern&, size_t) override {
    return absl::UnimplementedError("UpdateMemset is not implemented");
  }

  absl::StatusOr<const Command*> CreateDnnGraphCommand(
      se::dnn::DnnGraph&, se::Stream&, absl::Span<se::DeviceAddressBase>,
      absl::Span<const Command* const>) override {
    return absl::UnimplementedError("CreateDnnGraphCommand is not implemented");
  }

  absl::Status UpdateDnnGraphCommand(
      const Command*, se::dnn::DnnGraph&, se::Stream&,
      absl::Span<se::DeviceAddressBase>) override {
    return absl::UnimplementedError("UpdateDnnGraphCommand is not implemented");
  }

  absl::StatusOr<const Command*> CreateCase(
      se::DeviceAddress<int32_t>, std::vector<CreateCommands> create_branches,
      absl::Span<const Command* const> dependencies) override {
    return CreateCase(std::move(create_branches), dependencies);
  }

  absl::StatusOr<const Command*> CreateCase(
      se::DeviceAddress<bool>, std::vector<CreateCommands> create_branches,
      absl::Span<const Command* const> dependencies) override {
    return CreateCase(std::move(create_branches), dependencies);
  }

  absl::Status UpdateCase(
      const Command* command, se::DeviceAddress<int32_t>,
      std::vector<UpdateCommands> update_branches) override {
    return UpdateCase(command, std::move(update_branches));
  }

  absl::Status UpdateCase(
      const Command* command, se::DeviceAddress<bool>,
      std::vector<UpdateCommands> update_branches) override {
    return UpdateCase(command, std::move(update_branches));
  }

  absl::StatusOr<const Command*> CreateWhile(
      se::DeviceAddress<bool>, CreateCommands, CreateCommands,
      absl::Span<const Command* const>) override {
    return absl::UnimplementedError("CreateWhile is not implemented");
  }

  absl::Status UpdateWhile(const Command*, se::DeviceAddress<bool>,
                           UpdateCommands, UpdateCommands) override {
    return absl::UnimplementedError("UpdateWhile is not implemented");
  }

  absl::Status SetPriority(se::StreamPriority) override {
    return absl::OkStatus();
  }

  absl::Status Finalize() override {
    if (state_ != State::kCreate && state_ != State::kUpdate) {
      return absl::FailedPreconditionError(
          "command buffer is not in create/update state");
    }
    state_ = State::kFinalized;
    return absl::OkStatus();
  }

  absl::Status Update() override {
    if (state_ != State::kFinalized) {
      return absl::FailedPreconditionError(
          "command buffer is not in finalized state");
    }
    state_ = State::kUpdate;
    return absl::OkStatus();
  }

  Mode mode() const override { return mode_; }
  State state() const override { return state_; }
  std::string ToString() const override { return "FakeCommandBuffer"; }

 private:
  absl::Status Trace(se::Stream*, absl::AnyInvocable<absl::Status()>) override {
    return absl::UnimplementedError("Trace is not implemented");
  }

  absl::StatusOr<const Command*> CreateCase(
      std::vector<CreateCommands> create_branches,
      absl::Span<const Command* const> dependencies) {
    if (state_ != State::kCreate) {
      return absl::FailedPreconditionError(
          "command buffer is not in create state");
    }

    ++create_case_calls_;
    create_case_branch_count_ = create_branches.size();
    create_case_dependency_count_ = dependencies.size();
    branch_command_buffers_.clear();
    branch_command_buffers_.reserve(create_branches.size());

    for (CreateCommands& create_branch : create_branches) {
      auto branch_command_buffer =
          std::make_unique<FakeCommandBuffer>(Mode::kNested);
      TF_RETURN_IF_ERROR(create_branch(branch_command_buffer.get(),
                                       /*dependencies=*/{})
                             .status());
      TF_RETURN_IF_ERROR(branch_command_buffer->Finalize());
      branch_command_buffers_.push_back(std::move(branch_command_buffer));
    }

    return &case_command_;
  }

  absl::Status UpdateCase(const Command* command,
                          std::vector<UpdateCommands> update_branches) {
    if (state_ != State::kUpdate) {
      return absl::FailedPreconditionError(
          "command buffer is not in update state");
    }
    if (command != &case_command_) {
      return absl::InternalError("unexpected case command");
    }
    if (update_branches.size() != branch_command_buffers_.size()) {
      return absl::InternalError("unexpected branch count");
    }

    ++update_case_calls_;
    update_case_branch_count_ = update_branches.size();

    for (size_t i = 0; i < update_branches.size(); ++i) {
      TF_RETURN_IF_ERROR(branch_command_buffers_[i]->Update());
      TF_RETURN_IF_ERROR(update_branches[i](branch_command_buffers_[i].get()));
      TF_RETURN_IF_ERROR(branch_command_buffers_[i]->Finalize());
    }

    return absl::OkStatus();
  }

  Mode mode_;
  State state_ = State::kCreate;
  FakeSeCommand case_command_;
  FakeSeCommand dependency_command_;
  int create_case_calls_ = 0;
  int update_case_calls_ = 0;
  int create_case_branch_count_ = 0;
  int update_case_branch_count_ = 0;
  int create_case_dependency_count_ = 0;
  std::vector<std::unique_ptr<FakeCommandBuffer>> branch_command_buffers_;
};

TEST(ConditionalThunkTest, BufferUses) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);

  constexpr bool kBranchIndexIsBool = true;
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);
  Shape shape = ShapeUtil::MakeShape(PRED, {});

  ThunkSequence false_seq;
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  ThunkSequence true_seq;
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  std::vector<ThunkSequence> branch_thunk_sequences;
  branch_thunk_sequences.push_back(std::move(false_seq));
  branch_thunk_sequences.push_back(std::move(true_seq));

  std::unique_ptr<ConditionalThunk> thunk = CreateConditionalThunk(
      thunk_info, {slice, shape}, std::move(branch_thunk_sequences));

  EXPECT_EQ(thunk->branch_index_is_bool(), kBranchIndexIsBool);
  EXPECT_EQ(thunk->branch_index_buffer().slice, slice);

  auto thunk_matcher = Pointee(Property(&Thunk::kind, Thunk::Kind::kGemm));
  auto branch_matcher = Property(&ThunkExecutor::thunks,
                                 ElementsAre(thunk_matcher, thunk_matcher));
  EXPECT_THAT(thunk->branch_executors(),
              ElementsAre(branch_matcher, branch_matcher));
}

TEST(ConditionalThunkTest, RecordCreatesAndUpdatesCommandBufferCase) {
  BufferAllocation branch_index_alloc(/*index=*/0, /*size=*/sizeof(int32_t),
                                      /*color=*/0);
  BufferAllocation::Slice branch_index_slice(&branch_index_alloc, /*offset=*/0,
                                             /*size=*/sizeof(int32_t));

  std::vector<ThunkSequence> branch_thunks(2);
  ConditionalThunk thunk(
      Thunk::ThunkInfo(),
      ShapedSlice{branch_index_slice, ShapeUtil::MakeShape(S32, {})},
      std::move(branch_thunks));

  BranchRecordCounts branch0_counts;
  BranchRecordCounts branch1_counts;
  std::vector<CommandExecutor> branch_executors;
  ASSERT_OK_AND_ASSIGN(CommandExecutor branch0_executor,
                       MakeBranchExecutor(&branch0_counts));
  ASSERT_OK_AND_ASSIGN(CommandExecutor branch1_executor,
                       MakeBranchExecutor(&branch1_counts));
  branch_executors.push_back(std::move(branch0_executor));
  branch_executors.push_back(std::move(branch1_executor));
  ASSERT_OK(thunk.SetOrUpdateCommandBufferBranchExecutors(
      std::move(branch_executors)));

  int32_t branch_index = 0;
  std::vector<se::DeviceAddressBase> buffers = {
      se::DeviceAddressBase(&branch_index, sizeof(branch_index))};
  BufferAllocations allocations(buffers, /*device_ordinal=*/0,
                                /*memory_allocator=*/nullptr);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, allocations, /*stream=*/nullptr,
      /*command_buffer_trace_stream=*/nullptr, /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

  CommandStateManager state_manager;
  Command::RecordParams record_params{state_manager};
  FakeCommandBuffer command_buffer;

  std::vector<const se::CommandBuffer::Command*> dependencies = {
      command_buffer.dependency_command()};
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* case_command,
      thunk.Record(execute_params, record_params,
                   Command::RecordCreate{dependencies}, &command_buffer));

  EXPECT_EQ(case_command, command_buffer.case_command());
  EXPECT_EQ(command_buffer.create_case_calls(), 1);
  EXPECT_EQ(command_buffer.create_case_branch_count(), 2);
  EXPECT_EQ(command_buffer.create_case_dependency_count(), 1);
  EXPECT_EQ(branch0_counts.creates, 1);
  EXPECT_EQ(branch1_counts.creates, 1);
  EXPECT_EQ(branch0_counts.updates, 0);
  EXPECT_EQ(branch1_counts.updates, 0);

  ASSERT_OK(command_buffer.Finalize());
  ASSERT_OK(command_buffer.Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_case_command,
      thunk.Record(execute_params, record_params,
                   Command::RecordUpdate{case_command}, &command_buffer));

  EXPECT_EQ(updated_case_command, case_command);
  EXPECT_EQ(command_buffer.update_case_calls(), 1);
  EXPECT_EQ(command_buffer.update_case_branch_count(), 2);
  EXPECT_EQ(branch0_counts.creates, 1);
  EXPECT_EQ(branch1_counts.creates, 1);
  EXPECT_EQ(branch0_counts.updates, 1);
  EXPECT_EQ(branch1_counts.updates, 1);
}

TEST(ConditionalThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);
  Shape shape = ShapeUtil::MakeShape(PRED, {});

  ThunkSequence false_seq;
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  ThunkSequence true_seq;
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  std::vector<ThunkSequence> branch_thunk_seq;
  branch_thunk_seq.push_back(std::move(false_seq));
  branch_thunk_seq.push_back(std::move(true_seq));

  std::unique_ptr<ConditionalThunk> thunk = CreateConditionalThunk(
      thunk_info, {slice, shape}, std::move(branch_thunk_seq));
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk->ToProto());

  std::string expected = R"pb(
    thunk_info { profile_annotation: "profile_annotation" }
    conditional_thunk {
      branch_index_buffer {
        slice { size: 256 }
        shape {
          element_type: PRED
          layout { tail_padding_alignment_in_elements: 1 }
        }
      }
      branch_thunks {
        thunks { thunk_info { profile_annotation: "profile_annotation" } }
        thunks { thunk_info { profile_annotation: "profile_annotation" } }
      }
      branch_thunks {
        thunks { thunk_info { profile_annotation: "profile_annotation" } }
        thunks { thunk_info { profile_annotation: "profile_annotation" } }
      }
    }
  )pb";
  EXPECT_THAT(proto, EqualsProto(expected));
}

TEST(ConditionalThunkTest, FromProto) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        conditional_thunk {
          branch_index_buffer {
            slice { offset: 8 size: 256 buffer_allocation_index: 0 }
            shape {
              element_type: PRED
              layout { tail_padding_alignment_in_elements: 1 }
            }
          }
          branch_thunks {
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
          }
          branch_thunks {
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
            thunks { thunk_info { profile_annotation: "profile_annotation" } }
          }
        }
      )pb",
      &proto));

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ConditionalThunk> thunk,
      ConditionalThunk::FromProto(
          thunk_info, proto.conditional_thunk(), buffer_allocations,
          [](const ThunkProto& proto)
              -> absl::StatusOr<std::unique_ptr<DummyThunk>> {
            return DummyThunk::FromProto(proto, Kind::kCustomCall);
          }));
  ASSERT_NE(thunk, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(ConditionalThunkTest, ToString) {
  Thunk::ThunkInfo thunk_info;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/256);
  Shape bool_shape = ShapeUtil::MakeShape(PRED, {});
  Shape int_shape = ShapeUtil::MakeShape(S32, {});

  auto create_branch_thunk_sequences = [&]() -> std::vector<ThunkSequence> {
    ThunkSequence false_seq;
    false_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

    ThunkSequence true_seq;
    true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
    true_seq.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

    std::vector<ThunkSequence> branch_thunk_sequences;
    branch_thunk_sequences.push_back(std::move(false_seq));
    branch_thunk_sequences.push_back(std::move(true_seq));
    return branch_thunk_sequences;
  };

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(CreateConditionalThunk(
      thunk_info, {slice, bool_shape}, create_branch_thunk_sequences()));
  EXPECT_EQ(thunk_sequence.ToString(/*indent=*/0),
            "000: kConditional [source | sink]   \n"
            "  false_branch:\n"
            "    000: kGemm [source | sink] (no description)\n"
            "  true_branch:\n"
            "    000: kGemm [source | sink] (no description)\n"
            "    000: kGemm [source | sink] (no description)\n");

  std::unique_ptr<ConditionalThunk> thunk = CreateConditionalThunk(
      thunk_info, {slice, int_shape}, create_branch_thunk_sequences());

  EXPECT_EQ(thunk->ToString(/*indent=*/0),
            "\n"
            "branch_0:\n"
            "  000: kGemm [source | sink] (no description)\n"
            "branch_1:\n"
            "  000: kGemm [source | sink] (no description)\n"
            "  000: kGemm [source | sink] (no description)\n");
}

TEST(ConditionalThunkTest, TransformNested) {
  BufferAllocation::Slice slice;
  Shape shape = ShapeUtil::MakeShape(S32, {});
  Thunk::ThunkInfo thunk_info;

  ThunkSequence branch0;
  branch0.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));
  ThunkSequence branch1;
  branch1.push_back(std::make_unique<DummyThunk>(Kind::kGemm, thunk_info));

  std::vector<ThunkSequence> branch_thunks;
  branch_thunks.push_back(std::move(branch0));
  branch_thunks.push_back(std::move(branch1));
  auto conditional_thunk = std::make_unique<ConditionalThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice, shape}, std::move(branch_thunks));

  TF_EXPECT_OK(conditional_thunk->TransformNested([](auto) {
    return std::make_unique<DummyThunk>(Kind::kCustomCall, Thunk::ThunkInfo());
  }));

  EXPECT_THAT(conditional_thunk->branch_executors(), SizeIs(2));
  EXPECT_THAT(conditional_thunk->branch_executors()[0].thunks(), SizeIs(1));
  EXPECT_THAT(conditional_thunk->branch_executors()[0].thunks()[0]->kind(),
              Kind::kCustomCall);
  EXPECT_THAT(conditional_thunk->branch_executors()[1].thunks(), SizeIs(1));
  EXPECT_THAT(conditional_thunk->branch_executors()[1].thunks()[0]->kind(),
              Kind::kCustomCall);
}

}  // namespace
}  // namespace xla::gpu
