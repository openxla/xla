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

#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_fusion_v2_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/execution_graph.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

MATCHER_P(HasEdgeTo, node_id, "") { return arg.id == node_id; }

class FakeKernelThunk : public KernelThunk {
 public:
  FakeKernelThunk(Thunk::ThunkInfo thunk_info, BufferAllocation::Slice slice)
      : KernelThunk(thunk_info,
                    /*kernel_name=*/"kernel",
                    /*kernel_arguments=*/BuildKernelArguments(slice),
                    /*launch_dimensions=*/LaunchDimensions(),
                    /*cluster_dim=*/se::ClusterDim(),
                    /*shmem_bytes=*/0,
                    /*tma_metadata=*/se::gpu::TmaMetadata(),
                    /*zeroed_output_buffer_indices=*/std::vector<int64_t>{}) {}

 private:
  static emitters::KernelArguments BuildKernelArguments(
      BufferAllocation::Slice slice) {
    emitters::KernelArgument arg(ShapeUtil::MakeShape(F32, {256}), slice);
    arg.set_written(true);
    return emitters::KernelArguments(
        std::vector<emitters::KernelArgument>{arg});
  }
};

class CommandBufferCmdEmitterTest : public testing::Test {
 public:
  Thunk::ThunkInfo NextThunkInfo(absl::string_view profile_annotation) {
    Thunk::ThunkInfo thunk_info;
    thunk_info.thunk_id = thunk_id_generator_.GetNextThunkId();
    thunk_info.profile_annotation = profile_annotation;
    return thunk_info;
  }

  std::map<std::string, ExecutionGraph::NodeId> NamesToNodeIds(
      CommandExecutor& commands) {
    std::map<std::string, ExecutionGraph::NodeId> name_to_node_id;
    ExecutionGraph::NodeId node_id = 0;
    CHECK_OK(commands.Walk([&name_to_node_id, &node_id](Command* command) {
      name_to_node_id[std::string(command->profile_annotation())] = node_id++;
      return absl::OkStatus();
    }));
    return name_to_node_id;
  }

 private:
  ThunkIdGenerator thunk_id_generator_;
};

TEST_F(CommandBufferCmdEmitterTest, ConcurrentAndSequentialExecutionGraphs) {
  BufferAllocation allocation(/*index=*/0, /*size=*/2048, /*color=*/0);

  ThunkSequence thunks;
  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("a"), slice_a);
  BufferAllocation::Slice slice_b(&allocation, /*offset=*/1024, /*size=*/2048);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("b"), slice_b);

  ConvertToCommandsOptions options;
  {
    // Synchronization mode kConcurrent creates a concurrent execution graph.
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrent;
    ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                         ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
  }

  {
    // Synchronization mode kConcurrentRegions creates a sequential execution
    // graph when all thunks are in different concurrent regions.
    thunks[0]->set_concurrent_region_id(42);
    thunks[1]->set_concurrent_region_id(43);
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrentRegions;
    ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                         ConvertToCommands(thunks, options));
    EXPECT_TRUE(commands.execution_graph()->is_sequential());
  }

  {
    // Synchronization mode kConcurrentRegions creates a concurrent execution
    // graph when thunks are in the same regions.
    thunks[0]->set_concurrent_region_id(44);
    thunks[1]->set_concurrent_region_id(44);
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrentRegions;
    ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                         ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
  }
}

TEST_F(CommandBufferCmdEmitterTest,
       ConcurrentRegionsLimitesConcurrencyToTwoChains) {
  BufferAllocation allocation(/*index=*/0, /*size=*/3 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("a"), slice_a);

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/2 * 1024,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("b"), slice_b);

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("c"), slice_c);

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("d"), slice_d);

  ConvertToCommandsOptions options;
  // In kConcurrent mode, there are 3 source nodes (a, b, d) and 3 sink nodes
  // (c, b, d).
  {
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrent;
    ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                         ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
    EXPECT_EQ(commands.execution_graph()->nodes_defs().size(), 4);
    EXPECT_EQ(commands.execution_graph()->source().size(), 3);
    EXPECT_EQ(commands.execution_graph()->sink().size(), 3);
  }

  // In kConcurrentRegions mode and artificial dependencies between b and d is
  // inserted, hence there are 2 source nodes (a, b) and 2 sink nodes (c, d).
  {
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrentRegions;
    for (auto& thunk : thunks) {
      thunk->set_concurrent_region_id(44);
    }
    ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                         ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
    EXPECT_EQ(commands.execution_graph()->nodes_defs().size(), 4);
    EXPECT_EQ(commands.execution_graph()->source().size(), 2);
    EXPECT_EQ(commands.execution_graph()->sink().size(), 2);
    auto node_id = NamesToNodeIds(commands);

    EXPECT_EQ(node_id["a"], 0);
    EXPECT_EQ(node_id["b"], 1);
    EXPECT_EQ(node_id["c"], 2);
    EXPECT_EQ(node_id["d"], 3);

    EXPECT_EQ(
        commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges[0].id,
        node_id["c"]);
    EXPECT_EQ(
        commands.execution_graph()->nodes_defs()[node_id["b"]].out_edges[0].id,
        node_id["d"]);
  }
}

TEST_F(CommandBufferCmdEmitterTest, ConcurrentRegionsExecutesOutOfOrder) {
  BufferAllocation allocation(/*index=*/0, /*size=*/2 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("a"), slice_a);

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/0,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("b"), slice_b);

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("c"), slice_c);

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("d"), slice_d);

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  for (auto& thunk : thunks) {
    thunk->set_concurrent_region_id(44);
  }
  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options));
  EXPECT_FALSE(commands.execution_graph()->is_sequential());
  EXPECT_EQ(commands.execution_graph()->nodes_defs().size(), 4);
  EXPECT_EQ(commands.execution_graph()->source().size(), 2);
  EXPECT_EQ(commands.execution_graph()->sink().size(), 2);
  auto node_id = NamesToNodeIds(commands);

  // In the ThunkSequence the order is a, b, c, d. In the execution graph (a, c)
  // and (b, d) are executed concurrently. Hence, node c has a lower id than
  // node b.
  EXPECT_EQ(node_id["a"], 0);
  EXPECT_EQ(node_id["b"], 2);
  EXPECT_EQ(node_id["c"], 1);
  EXPECT_EQ(node_id["d"], 3);

  EXPECT_EQ(
      commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges[0].id,
      node_id["b"]);
  EXPECT_EQ(
      commands.execution_graph()->nodes_defs()[node_id["c"]].out_edges[0].id,
      node_id["d"]);
}

TEST_F(CommandBufferCmdEmitterTest,
       ConcurrentRegionsAddsDependenciesBetweenRegions) {
  BufferAllocation allocation(/*index=*/0, /*size=*/5 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("a"), slice_a);
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("b"), slice_b);
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/2 * 1024,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("c"), slice_c);
  thunks.back()->set_concurrent_region_id(43);

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/3 * 1024,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("d"), slice_d);
  thunks.back()->set_concurrent_region_id(43);

  BufferAllocation::Slice slice_e(&allocation, /*offset=*/4 * 1024,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("e"), slice_e);
  thunks.back()->set_concurrent_region_id(44);

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options));
  // The execution graph contains dependencies between nodes in concurrent
  // regions: (a, b), (c, d) and (e).
  EXPECT_FALSE(commands.execution_graph()->is_sequential());
  auto node_id = NamesToNodeIds(commands);
  EXPECT_THAT(
      commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges,
      UnorderedElementsAre(HasEdgeTo(node_id["c"]), HasEdgeTo(node_id["d"])));
  EXPECT_THAT(
      commands.execution_graph()->nodes_defs()[node_id["b"]].out_edges,
      UnorderedElementsAre(HasEdgeTo(node_id["c"]), HasEdgeTo(node_id["d"])));
  EXPECT_THAT(commands.execution_graph()->nodes_defs()[node_id["c"]].out_edges,
              UnorderedElementsAre(HasEdgeTo(node_id["e"])));
  EXPECT_THAT(commands.execution_graph()->nodes_defs()[node_id["d"]].out_edges,
              UnorderedElementsAre(HasEdgeTo(node_id["e"])));
}

TEST_F(CommandBufferCmdEmitterTest,
       ConcurrentRegionsAddsDependenciesBetweenRegionsDifferenLaneLength) {
  BufferAllocation allocation(/*index=*/0, /*size=*/5 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("a"), slice_a);
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("b"), slice_b);
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/2 * 1024,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("c"), slice_c);
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/3 * 1024,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("d"), slice_d);
  thunks.back()->set_concurrent_region_id(43);

  BufferAllocation::Slice slice_e(&allocation, /*offset=*/4 * 1024,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("e"), slice_e);
  thunks.back()->set_concurrent_region_id(43);

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options));
  // The execution graph contains dependencies between nodes in concurrent
  // regions: (a, b, c), (d, e).
  EXPECT_FALSE(commands.execution_graph()->is_sequential());
  auto node_id = NamesToNodeIds(commands);
  EXPECT_THAT(
      commands.execution_graph()->nodes_defs()[node_id["b"]].out_edges,
      UnorderedElementsAre(HasEdgeTo(node_id["d"]), HasEdgeTo(node_id["e"])));
  EXPECT_THAT(
      commands.execution_graph()->nodes_defs()[node_id["c"]].out_edges,
      UnorderedElementsAre(HasEdgeTo(node_id["d"]), HasEdgeTo(node_id["e"])));
}

TEST_F(CommandBufferCmdEmitterTest, ConcurrentRegionsScheduleHasLaneAffinity) {
  BufferAllocation allocation(/*index=*/0, /*size=*/3 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("a"), slice_a);

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("b"), slice_b);

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/2 * 1024,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("c"), slice_c);

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/3 * 1024,
                                  /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("d"), slice_d);

  BufferAllocation::Slice slice_e(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.Emplace<FakeKernelThunk>(NextThunkInfo("e"), slice_e);
  for (auto& thunk : thunks) {
    thunk->set_concurrent_region_id(44);
  }

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options));
  auto node_id = NamesToNodeIds(commands);
  EXPECT_EQ(node_id["a"], 0);
  EXPECT_EQ(node_id["b"], 1);
  // Nodes c and d are ready from the start. However, node e is scheduled ahead
  // of them, because it has a data dependency on a.
  EXPECT_EQ(node_id["e"], 2);

  // Nade e depends on a only and should be scheduled on the same lane as a.
  EXPECT_EQ(
      commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges.size(),
      1);
  EXPECT_EQ(
      commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges[0].id,
      node_id["e"]);
}

TEST_F(CommandBufferCmdEmitterTest, ConvertsConditionalThunkToCommand) {
  BufferAllocation branch_index_alloc(/*index=*/0, /*size=*/sizeof(int32_t),
                                      /*color=*/0);
  BufferAllocation data_alloc(/*index=*/1, /*size=*/2 * 1024, /*color=*/0);

  BufferAllocation::Slice branch_index_slice(&branch_index_alloc, /*offset=*/0,
                                             /*size=*/sizeof(int32_t));
  BufferAllocation::Slice branch0_slice(&data_alloc, /*offset=*/0,
                                        /*size=*/1024);
  BufferAllocation::Slice branch1_slice(&data_alloc, /*offset=*/1024,
                                        /*size=*/1024);

  ThunkSequence branch0 = ThunkSequence::Of<FakeKernelThunk>(
      NextThunkInfo("branch0"), branch0_slice);

  ThunkSequence branch1 = ThunkSequence::Of<FakeKernelThunk>(
      NextThunkInfo("branch1"), branch1_slice);

  std::vector<ThunkSequence> branches;
  branches.push_back(std::move(branch0));
  branches.push_back(std::move(branch1));

  auto conditional = std::make_unique<ConditionalThunk>(
      NextThunkInfo("conditional"),
      ShapedSlice{branch_index_slice, ShapeUtil::MakeShape(S32, {})},
      std::move(branches));
  ConditionalThunk* conditional_ptr = conditional.get();

  ThunkSequence thunks;
  thunks.push_back(std::move(conditional));

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, ConvertToCommandsOptions()));

  EXPECT_EQ(conditional_ptr->kind(), Thunk::Kind::kConditional);

  std::vector<std::string> command_names;
  CHECK_OK(commands.Walk([&](Command* command) {
    command_names.push_back(std::string(command->profile_annotation()));
    return absl::OkStatus();
  }));
  EXPECT_THAT(command_names, ElementsAre("conditional", "branch0", "branch1"));
}

TEST_F(CommandBufferCmdEmitterTest, ConvertsWhileThunkToCommand) {
  BufferAllocation pred_alloc(/*index=*/0, /*size=*/sizeof(bool), /*color=*/0);
  BufferAllocation data_alloc(/*index=*/1, /*size=*/2 * 1024, /*color=*/0);

  BufferAllocation::Slice pred_slice(&pred_alloc, /*offset=*/0,
                                     /*size=*/sizeof(bool));
  BufferAllocation::Slice cond_slice(&data_alloc, /*offset=*/0,
                                     /*size=*/1024);
  BufferAllocation::Slice body_slice(&data_alloc, /*offset=*/1024,
                                     /*size=*/1024);

  ThunkSequence cond_thunks =
      ThunkSequence::Of<FakeKernelThunk>(NextThunkInfo("cond"), cond_slice);

  ThunkSequence body_thunks =
      ThunkSequence::Of<FakeKernelThunk>(NextThunkInfo("body"), body_slice);

  auto while_thunk = std::make_unique<WhileThunk>(
      NextThunkInfo("while"), pred_slice, std::move(cond_thunks),
      std::move(body_thunks));
  WhileThunk* while_ptr = while_thunk.get();

  ThunkSequence thunks;
  thunks.push_back(std::move(while_thunk));

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, ConvertToCommandsOptions()));

  EXPECT_EQ(while_ptr->kind(), Thunk::Kind::kWhile);

  std::vector<std::string> command_names;
  CHECK_OK(commands.Walk([&](Command* command) {
    command_names.push_back(std::string(command->profile_annotation()));
    return absl::OkStatus();
  }));
  EXPECT_THAT(command_names, ElementsAre("while", "cond", "body"));
}

TEST_F(CommandBufferCmdEmitterTest, ConvertsWhileThunkRepeatedly) {
  BufferAllocation pred_alloc(/*index=*/0, /*size=*/sizeof(bool), /*color=*/0);
  BufferAllocation data_alloc(/*index=*/1, /*size=*/2 * 1024, /*color=*/0);

  BufferAllocation::Slice pred_slice(&pred_alloc, /*offset=*/0,
                                     /*size=*/sizeof(bool));
  BufferAllocation::Slice cond_slice(&data_alloc, /*offset=*/0,
                                     /*size=*/1024);
  BufferAllocation::Slice body_slice(&data_alloc, /*offset=*/1024,
                                     /*size=*/1024);

  ThunkSequence cond_thunks =
      ThunkSequence::Of<FakeKernelThunk>(NextThunkInfo("cond"), cond_slice);

  ThunkSequence body_thunks =
      ThunkSequence::Of<FakeKernelThunk>(NextThunkInfo("body"), body_slice);

  ThunkSequence thunks = ThunkSequence::Of<WhileThunk>(
      NextThunkInfo("while"), pred_slice, std::move(cond_thunks),
      std::move(body_thunks));

  auto collect_command_names = [](CommandExecutor& commands) {
    std::vector<std::string> command_names;
    CHECK_OK(commands.Walk([&](Command* command) {
      command_names.push_back(std::string(command->profile_annotation()));
      return absl::OkStatus();
    }));
    return command_names;
  };

  ASSERT_OK_AND_ASSIGN(CommandExecutor first_commands,
                       ConvertToCommands(thunks, ConvertToCommandsOptions()));
  EXPECT_THAT(collect_command_names(first_commands),
              ElementsAre("while", "cond", "body"));

  ConvertToCommandsOptions concurrent_options;
  concurrent_options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrent;
  ASSERT_OK_AND_ASSIGN(CommandExecutor second_commands,
                       ConvertToCommands(thunks, concurrent_options));
  ASSERT_TRUE(second_commands.execution_graph().has_value());
  EXPECT_THAT(collect_command_names(second_commands),
              ElementsAre("while", "cond", "body"));
}

TEST_F(CommandBufferCmdEmitterTest, ConvertsConditionalThunkRepeatedly) {
  BufferAllocation branch_index_alloc(/*index=*/0, /*size=*/sizeof(int32_t),
                                      /*color=*/0);
  BufferAllocation data_alloc(/*index=*/1, /*size=*/2 * 1024, /*color=*/0);

  BufferAllocation::Slice branch_index_slice(&branch_index_alloc, /*offset=*/0,
                                             /*size=*/sizeof(int32_t));
  BufferAllocation::Slice branch0_slice(&data_alloc, /*offset=*/0,
                                        /*size=*/1024);
  BufferAllocation::Slice branch1_slice(&data_alloc, /*offset=*/1024,
                                        /*size=*/1024);

  ThunkSequence branch0 = ThunkSequence::Of<FakeKernelThunk>(
      NextThunkInfo("branch0"), branch0_slice);

  ThunkSequence branch1 = ThunkSequence::Of<FakeKernelThunk>(
      NextThunkInfo("branch1"), branch1_slice);

  std::vector<ThunkSequence> branches;
  branches.push_back(std::move(branch0));
  branches.push_back(std::move(branch1));

  ThunkSequence thunks = ThunkSequence::Of<ConditionalThunk>(
      NextThunkInfo("conditional"),
      ShapedSlice{branch_index_slice, ShapeUtil::MakeShape(S32, {})},
      std::move(branches));

  auto collect_command_names = [](CommandExecutor& commands) {
    std::vector<std::string> command_names;
    CHECK_OK(commands.Walk([&](Command* command) {
      command_names.push_back(std::string(command->profile_annotation()));
      return absl::OkStatus();
    }));
    return command_names;
  };

  ASSERT_OK_AND_ASSIGN(CommandExecutor first_commands,
                       ConvertToCommands(thunks, ConvertToCommandsOptions()));
  EXPECT_THAT(collect_command_names(first_commands),
              ElementsAre("conditional", "branch0", "branch1"));

  ConvertToCommandsOptions concurrent_options;
  concurrent_options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrent;
  ASSERT_OK_AND_ASSIGN(CommandExecutor second_commands,
                       ConvertToCommands(thunks, concurrent_options));
  ASSERT_TRUE(second_commands.execution_graph().has_value());
  EXPECT_THAT(collect_command_names(second_commands),
              ElementsAre("conditional", "branch0", "branch1"));
}

TEST_F(CommandBufferCmdEmitterTest,
       ConvertsBoolConditionalBranchesInCaseOrder) {
  BufferAllocation branch_index_alloc(/*index=*/0, /*size=*/sizeof(bool),
                                      /*color=*/0);
  BufferAllocation data_alloc(/*index=*/1, /*size=*/2 * 1024, /*color=*/0);

  BufferAllocation::Slice branch_index_slice(&branch_index_alloc, /*offset=*/0,
                                             /*size=*/sizeof(bool));
  BufferAllocation::Slice false_slice(&data_alloc, /*offset=*/0, /*size=*/1024);
  BufferAllocation::Slice true_slice(&data_alloc, /*offset=*/1024,
                                     /*size=*/1024);

  ThunkSequence false_branch = ThunkSequence::Of<FakeKernelThunk>(
      NextThunkInfo("false_branch"), false_slice);

  ThunkSequence true_branch = ThunkSequence::Of<FakeKernelThunk>(
      NextThunkInfo("true_branch"), true_slice);

  std::vector<ThunkSequence> branches;
  branches.push_back(std::move(false_branch));
  branches.push_back(std::move(true_branch));

  ThunkSequence thunks = ThunkSequence::Of<ConditionalThunk>(
      NextThunkInfo("conditional"),
      ShapedSlice{branch_index_slice, ShapeUtil::MakeShape(PRED, {})},
      std::move(branches));

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, ConvertToCommandsOptions()));

  std::vector<std::string> command_names;
  CHECK_OK(commands.Walk([&](Command* command) {
    command_names.push_back(std::string(command->profile_annotation()));
    return absl::OkStatus();
  }));
  EXPECT_THAT(command_names,
              ElementsAre("conditional", "true_branch", "false_branch"));
}

TEST_F(CommandBufferCmdEmitterTest, ConcurrentRegionsNestedThunk) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&allocation, /*offset=*/0, /*size=*/1024);

  ThunkSequence nested_thunks =
      ThunkSequence::Of<FakeKernelThunk>(NextThunkInfo("nested_kernel"), slice);
  auto seq_thunk = std::make_unique<SequentialThunk>(NextThunkInfo("seq_thunk"),
                                                     std::move(nested_thunks));
  seq_thunk->set_concurrent_region_id(42);
  ThunkSequence thunks;
  thunks.push_back(std::move(seq_thunk));
  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  EXPECT_OK(ConvertToCommands(thunks, options));
}

TEST_F(CommandBufferCmdEmitterTest, ConcurrentRegionsWithEmptyThunkAtStart) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&allocation, /*offset=*/0, /*size=*/1024);
  ThunkSequence thunks;
  auto empty_seq = std::make_unique<SequentialThunk>(NextThunkInfo("empty_seq"),
                                                     ThunkSequence());
  empty_seq->set_concurrent_region_id(42);
  thunks.push_back(std::move(empty_seq));

  auto real_kernel =
      std::make_unique<FakeKernelThunk>(NextThunkInfo("real_kernel"), slice);
  real_kernel->set_concurrent_region_id(43);
  thunks.push_back(std::move(real_kernel));
  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options));
  EXPECT_EQ(commands.size(), 1);
  auto node_id = NamesToNodeIds(commands);
  EXPECT_EQ(node_id.size(), 1);
  EXPECT_EQ(node_id.count("real_kernel"), 1);
  EXPECT_EQ(node_id["real_kernel"], 0);
  ASSERT_TRUE(commands.execution_graph().has_value());
  EXPECT_THAT(commands.execution_graph()->nodes_defs(),
              ElementsAre(AllOf(
                  Field(&ExecutionGraph::NodeDef::in_edges, IsEmpty()),
                  Field(&ExecutionGraph::NodeDef::out_edges, IsEmpty()))));
}

TEST_F(CommandBufferCmdEmitterTest, ConcurrentRegionsNestedThunksMultiple) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&allocation, /*offset=*/0, /*size=*/1024);

  ThunkSequence nested_thunks_1;
  nested_thunks_1.Emplace<FakeKernelThunk>(NextThunkInfo("k1"), slice);
  nested_thunks_1.Emplace<FakeKernelThunk>(NextThunkInfo("k2"), slice);
  auto seq_thunk_1 = std::make_unique<SequentialThunk>(
      NextThunkInfo("seq1"), std::move(nested_thunks_1));
  seq_thunk_1->set_concurrent_region_id(42);

  ThunkSequence nested_thunks_2;
  nested_thunks_2.Emplace<FakeKernelThunk>(NextThunkInfo("k3"), slice);
  nested_thunks_2.Emplace<FakeKernelThunk>(NextThunkInfo("k4"), slice);
  auto seq_thunk_2 = std::make_unique<SequentialThunk>(
      NextThunkInfo("seq2"), std::move(nested_thunks_2));
  seq_thunk_2->set_concurrent_region_id(42);

  ThunkSequence thunks;
  thunks.push_back(std::move(seq_thunk_1));
  thunks.push_back(std::move(seq_thunk_2));

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options));

  std::map<std::string, ExecutionGraph::NodeId> name_to_ids =
      NamesToNodeIds(commands);
  EXPECT_EQ(name_to_ids["k1"], 0);
  EXPECT_EQ(name_to_ids["k2"], 1);
  EXPECT_EQ(name_to_ids["k3"], 2);
  EXPECT_EQ(name_to_ids["k4"], 3);

  // "k3" (first command of successor seq_thunk_2) should depend on "k2"
  // (last command of predecessor seq_thunk_1).
  EXPECT_THAT(
      commands.execution_graph()->nodes_defs()[name_to_ids["k2"]].out_edges,
      UnorderedElementsAre(HasEdgeTo(name_to_ids["k3"])));
}

TEST_F(CommandBufferCmdEmitterTest,
       ConcurrentRegionsBoundarySafetyWithEmptyThunks) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/512);
  BufferAllocation::Slice slice_b(&allocation, /*offset=*/512, /*size=*/512);
  ThunkSequence thunks;

  const auto add_real_thunk = [&](absl::string_view name,
                                  BufferAllocation::Slice slice,
                                  uint64_t region_id) {
    auto real_thunk =
        std::make_unique<FakeKernelThunk>(NextThunkInfo(name), slice);
    real_thunk->set_concurrent_region_id(region_id);
    thunks.push_back(std::move(real_thunk));
  };
  const auto add_empty_thunk = [&](absl::string_view name, uint64_t region_id) {
    auto empty_thunk =
        std::make_unique<SequentialThunk>(NextThunkInfo(name), ThunkSequence());
    empty_thunk->set_concurrent_region_id(region_id);
    thunks.push_back(std::move(empty_thunk));
  };
  // Expected schedule (with 2 lanes):
  // Region0: real_a, real_b, empty_c
  //    Sources: real_a, real_b
  //    Sinks: empty_c, real_b
  //      empty_c
  //      real_b
  //  Region1: empty_d, real_f, real_e
  //    Sources: empty_d, real_f
  //    Sinks: real_e, real_f
  add_real_thunk("real_a", slice_a, 42);
  add_real_thunk("real_b", slice_b, 42);
  add_empty_thunk("empty_c", 42);  // < Empty sink.
  add_empty_thunk("empty_d", 43);  // < Empty source.
  add_real_thunk("real_f", slice_b, 43);
  add_real_thunk("real_e", slice_a, 43);

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options));

  std::map<std::string, ExecutionGraph::NodeId> name_to_ids =
      NamesToNodeIds(commands);
  // Safely store the temporary to avoid the dangling C++ span reference!
  std::optional<const ExecutionGraph> graph = commands.execution_graph();
  ASSERT_TRUE(graph.has_value());
  absl::Span<const ExecutionGraph::NodeDef> nodes = graph->nodes_defs();

  ASSERT_NE(name_to_ids.find("real_a"), name_to_ids.end());
  ASSERT_NE(name_to_ids.find("real_b"), name_to_ids.end());
  ASSERT_NE(name_to_ids.find("real_e"), name_to_ids.end());
  ASSERT_NE(name_to_ids.find("real_f"), name_to_ids.end());

  EXPECT_THAT(nodes[name_to_ids["real_a"]].out_edges,
              UnorderedElementsAre(HasEdgeTo(name_to_ids["real_e"]),
                                   HasEdgeTo(name_to_ids["real_f"])));
  EXPECT_THAT(nodes[name_to_ids["real_b"]].out_edges,
              UnorderedElementsAre(HasEdgeTo(name_to_ids["real_e"]),
                                   HasEdgeTo(name_to_ids["real_f"])));
}

TEST_F(CommandBufferCmdEmitterTest,
       ConcurrentRegionsFlatteningEnforcesLaneLimit) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/256);
  BufferAllocation::Slice slice_b(&allocation, /*offset=*/256, /*size=*/256);
  BufferAllocation::Slice slice_c(&allocation, /*offset=*/512, /*size=*/256);
  ThunkSequence thunks;
  auto k1 = std::make_unique<FakeKernelThunk>(NextThunkInfo("k1"), slice_a);
  k1->set_concurrent_region_id(42);
  thunks.push_back(std::move(k1));
  // seq_thunk (42) contains k2 (slice_b, 42) and k3 (slice_c, 42)
  ThunkSequence nested_thunks;
  nested_thunks.Emplace<FakeKernelThunk>(NextThunkInfo("k2"), slice_b);
  nested_thunks.Emplace<FakeKernelThunk>(NextThunkInfo("k3"), slice_c);
  auto seq_thunk = std::make_unique<SequentialThunk>(NextThunkInfo("seq"),
                                                     std::move(nested_thunks));
  seq_thunk->set_concurrent_region_id(42);
  thunks.push_back(std::move(seq_thunk));
  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options));

  std::map<std::string, ExecutionGraph::NodeId> name_to_ids =
      NamesToNodeIds(commands);
  std::optional<const ExecutionGraph> graph = commands.execution_graph();
  ASSERT_TRUE(graph.has_value());
  absl::Span<const ExecutionGraph::NodeDef> nodes = graph->nodes_defs();

  ASSERT_NE(name_to_ids.find("k1"), name_to_ids.end());
  ASSERT_NE(name_to_ids.find("k2"), name_to_ids.end());
  ASSERT_NE(name_to_ids.find("k3"), name_to_ids.end());

  // Since k1, k2, and k3 have no data conflicts, they are scheduled as:
  // Lane 0: k1 -> k3
  // Lane 1: k2
  EXPECT_THAT(nodes[name_to_ids["k1"]].out_edges,
              UnorderedElementsAre(HasEdgeTo(name_to_ids["k3"])));
  EXPECT_THAT(nodes[name_to_ids["k2"]].out_edges, IsEmpty());
}

TEST_F(CommandBufferCmdEmitterTest, ConcurrentRegionsForwardDependencies) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/256);
  BufferAllocation::Slice slice_b(&allocation, /*offset=*/256, /*size=*/256);
  BufferAllocation::Slice slice_c(&allocation, /*offset=*/512, /*size=*/256);
  BufferAllocation::Slice slice_d(&allocation, /*offset=*/768, /*size=*/256);
  ThunkSequence thunks;

  const auto add_kernel = [&](absl::string_view name,
                              BufferAllocation::Slice slice) {
    auto kernel = std::make_unique<FakeKernelThunk>(NextThunkInfo(name), slice);
    kernel->set_concurrent_region_id(42);
    thunks.push_back(std::move(kernel));
  };

  add_kernel("a", slice_a);
  add_kernel("b", slice_b);
  add_kernel("c", slice_c);
  add_kernel("d", slice_d);

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options));

  std::map<std::string, ExecutionGraph::NodeId> name_to_ids =
      NamesToNodeIds(commands);
  std::optional<const ExecutionGraph> graph = commands.execution_graph();
  ASSERT_TRUE(graph.has_value());
  absl::Span<const ExecutionGraph::NodeDef> nodes = graph->nodes_defs();

  ASSERT_NE(name_to_ids.find("a"), name_to_ids.end());
  ASSERT_NE(name_to_ids.find("b"), name_to_ids.end());
  ASSERT_NE(name_to_ids.find("c"), name_to_ids.end());
  ASSERT_NE(name_to_ids.find("d"), name_to_ids.end());

  // Since a, b, c, and d have no data conflicts, they are scheduled as:
  // Lane 0: a -> c
  // Lane 1: b -> d
  EXPECT_THAT(nodes[name_to_ids["a"]].out_edges,
              UnorderedElementsAre(HasEdgeTo(name_to_ids["c"])));
  EXPECT_THAT(nodes[name_to_ids["b"]].out_edges,
              UnorderedElementsAre(HasEdgeTo(name_to_ids["d"])));
  EXPECT_THAT(nodes[name_to_ids["c"]].out_edges, IsEmpty());
  EXPECT_THAT(nodes[name_to_ids["d"]].out_edges, IsEmpty());
}

class AsyncCommandBufferCmdEmitterTest : public CommandBufferCmdEmitterTest {
 protected:
  void Kernel(ThunkSequence& thunks, absl::string_view name, int64_t slot) {
    thunks.Emplace<FakeKernelThunk>(
        NextThunkInfo(name),
        BufferAllocation::Slice(&allocation_, slot * 1024, 1024));
  }

  AsyncStartThunk* Start(ThunkSequence& thunks, ExecutionStreamId stream,
                         ThunkSequence body) {
    auto start = std::make_unique<AsyncStartThunk>(NextThunkInfo("start"),
                                                   stream, std::move(body));
    AsyncStartThunk* result = start.get();
    thunks.push_back(std::move(start));
    return result;
  }

  void Done(ThunkSequence& thunks, const AsyncStartThunk* start,
            absl::string_view name) {
    thunks.Emplace<AsyncDoneThunk>(NextThunkInfo(name),
                                   start->async_execution());
  }

  // Check reachability rather than direct edges: graph construction can remove
  // redundant edges without changing the execution order.
  bool HappensBefore(CommandExecutor& commands, const std::string& before,
                     const std::string& after) {
    auto ids = NamesToNodeIds(commands);
    if (ids.find(before) == ids.end() || ids.find(after) == ids.end()) {
      return false;
    }
    auto graph = commands.execution_graph();
    std::vector<bool> reachable(commands.size(), false);
    reachable[ids.at(before)] = true;
    for (int64_t i = ids.at(before); i < static_cast<int64_t>(commands.size());
         ++i) {
      if (!reachable[i]) {
        continue;
      }
      for (const auto& edge : graph->nodes_defs()[i].out_edges) {
        reachable[edge.id] = true;
      }
    }
    return reachable[ids.at(after)];
  }

  const ConvertToCommandsOptions options_ = {
      CommandExecutor::SynchronizationMode::kLHS};

 private:
  BufferAllocation allocation_{0, 16 * 1024, 0};
};

TEST_F(AsyncCommandBufferCmdEmitterTest, PreservesForkBodyOrderAndJoin) {
  ThunkSequence thunks;
  Kernel(thunks, "before", 0);
  ThunkSequence body;
  Kernel(body, "async_a", 1);
  ThunkSequence nested;
  Kernel(nested, "async_b", 2);
  body.Emplace<SequentialThunk>(NextThunkInfo("sequential"), std::move(nested));
  AsyncStartThunk* start =
      Start(thunks, CommunicationStreamId(0), std::move(body));
  Kernel(thunks, "main_a", 3);
  Kernel(thunks, "main_b", 4);
  Done(thunks, start, "done");
  Kernel(thunks, "after", 5);

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options_));
  EXPECT_EQ(commands.size(), 7);
  EXPECT_TRUE(HappensBefore(commands, "before", "async_a"));
  EXPECT_TRUE(HappensBefore(commands, "before", "main_a"));
  EXPECT_TRUE(HappensBefore(commands, "async_a", "async_b"));
  EXPECT_TRUE(HappensBefore(commands, "main_a", "main_b"));
  EXPECT_TRUE(HappensBefore(commands, "async_b", "done"));
  EXPECT_TRUE(HappensBefore(commands, "main_b", "done"));
  EXPECT_TRUE(HappensBefore(commands, "done", "after"));
  EXPECT_FALSE(HappensBefore(commands, "async_b", "main_a"));
  EXPECT_FALSE(HappensBefore(commands, "main_b", "async_a"));
}

TEST_F(AsyncCommandBufferCmdEmitterTest,
       SameStreamOrdersRegionsButJoinsSnapshot) {
  ThunkSequence thunks;
  ThunkSequence first;
  Kernel(first, "first", 0);
  AsyncStartThunk* start_a =
      Start(thunks, ComputationStreamId(0), std::move(first));
  ThunkSequence second;
  Kernel(second, "second", 1);
  AsyncStartThunk* start_b =
      Start(thunks, ComputationStreamId(0), std::move(second));
  Done(thunks, start_a, "done_a");
  Kernel(thunks, "main", 2);
  Done(thunks, start_b, "done_b");

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options_));
  ASSERT_EQ(commands.size(), 5);
  EXPECT_TRUE(HappensBefore(commands, "first", "second"));
  EXPECT_TRUE(HappensBefore(commands, "first", "done_a"));
  EXPECT_TRUE(HappensBefore(commands, "done_a", "main"));
  EXPECT_TRUE(HappensBefore(commands, "second", "done_b"));
  EXPECT_TRUE(HappensBefore(commands, "main", "done_b"));
  EXPECT_FALSE(HappensBefore(commands, "second", "done_a"));
  EXPECT_FALSE(HappensBefore(commands, "main", "second"));
}

TEST_F(AsyncCommandBufferCmdEmitterTest,
       DistinguishesComputeAndCommunicationStreams) {
  ThunkSequence thunks;
  ThunkSequence compute;
  Kernel(compute, "compute", 0);
  AsyncStartThunk* start_a =
      Start(thunks, ComputationStreamId(0), std::move(compute));
  ThunkSequence communication;
  Kernel(communication, "communication", 1);
  AsyncStartThunk* start_b =
      Start(thunks, CommunicationStreamId(0), std::move(communication));
  Done(thunks, start_a, "done_a");
  Done(thunks, start_b, "done_b");

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options_));
  ASSERT_EQ(commands.size(), 4);
  EXPECT_FALSE(HappensBefore(commands, "compute", "communication"));
  EXPECT_FALSE(HappensBefore(commands, "communication", "compute"));
  EXPECT_TRUE(HappensBefore(commands, "compute", "done_a"));
  EXPECT_TRUE(HappensBefore(commands, "communication", "done_b"));
  EXPECT_TRUE(HappensBefore(commands, "done_a", "done_b"));
}

TEST_F(AsyncCommandBufferCmdEmitterTest, PreservesNestedAsyncRegions) {
  ThunkSequence thunks;
  ThunkSequence outer;
  Kernel(outer, "outer_before", 0);
  ThunkSequence inner;
  Kernel(inner, "inner", 1);
  AsyncStartThunk* inner_start =
      Start(outer, CommunicationStreamId(0), std::move(inner));
  Kernel(outer, "outer_during", 2);
  Done(outer, inner_start, "inner_done");
  Kernel(outer, "outer_after", 3);
  AsyncStartThunk* outer_start =
      Start(thunks, ComputationStreamId(0), std::move(outer));
  Kernel(thunks, "main", 4);
  Done(thunks, outer_start, "outer_done");

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options_));
  ASSERT_EQ(commands.size(), 7);
  EXPECT_TRUE(HappensBefore(commands, "outer_before", "inner"));
  EXPECT_TRUE(HappensBefore(commands, "outer_before", "outer_during"));
  EXPECT_FALSE(HappensBefore(commands, "inner", "outer_during"));
  EXPECT_TRUE(HappensBefore(commands, "inner", "inner_done"));
  EXPECT_TRUE(HappensBefore(commands, "outer_during", "inner_done"));
  EXPECT_TRUE(HappensBefore(commands, "inner_done", "outer_after"));
  EXPECT_TRUE(HappensBefore(commands, "outer_after", "outer_done"));
  EXPECT_TRUE(HappensBefore(commands, "main", "outer_done"));
  EXPECT_FALSE(HappensBefore(commands, "outer_after", "main"));
}

TEST_F(AsyncCommandBufferCmdEmitterTest, EmptyRegionCarriesStreamWaits) {
  ThunkSequence thunks;
  ThunkSequence first;
  Kernel(first, "first", 3);
  AsyncStartThunk* first_start =
      Start(thunks, ComputationStreamId(0), std::move(first));
  Kernel(thunks, "before", 0);
  AsyncStartThunk* empty = Start(thunks, ComputationStreamId(0), {});
  ThunkSequence body;
  Kernel(body, "async", 1);
  AsyncStartThunk* start =
      Start(thunks, ComputationStreamId(0), std::move(body));
  Done(thunks, empty, "empty_done");
  Kernel(thunks, "main", 2);
  Done(thunks, start, "done");
  Done(thunks, first_start, "first_done");

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options_));
  ASSERT_EQ(commands.size(), 7);
  EXPECT_FALSE(HappensBefore(commands, "first", "before"));
  EXPECT_TRUE(HappensBefore(commands, "first", "async"));
  EXPECT_TRUE(HappensBefore(commands, "first", "empty_done"));
  EXPECT_TRUE(HappensBefore(commands, "before", "async"));
  EXPECT_TRUE(HappensBefore(commands, "before", "empty_done"));
  EXPECT_TRUE(HappensBefore(commands, "empty_done", "main"));
  EXPECT_FALSE(HappensBefore(commands, "async", "empty_done"));
  EXPECT_TRUE(HappensBefore(commands, "async", "done"));
  EXPECT_TRUE(HappensBefore(commands, "main", "done"));
}

TEST_F(AsyncCommandBufferCmdEmitterTest, MatchesSharedAsyncExecution) {
  ThunkSequence thunks;
  ThunkSequence first;
  Kernel(first, "first", 0);
  AsyncStartThunk* canonical =
      Start(thunks, CommunicationStreamId(0), std::move(first));
  Done(thunks, canonical, "first_done");
  ThunkSequence second;
  Kernel(second, "second", 1);
  thunks.Emplace<AsyncStartThunk>(NextThunkInfo("pipelined_start"),
                                  CommunicationStreamId(0), std::move(second),
                                  canonical->async_execution());
  Done(thunks, canonical, "second_done");

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options_));
  ASSERT_EQ(commands.size(), 4);
  EXPECT_TRUE(HappensBefore(commands, "first", "first_done"));
  EXPECT_TRUE(HappensBefore(commands, "first_done", "second"));
  EXPECT_TRUE(HappensBefore(commands, "second", "second_done"));
}

TEST_F(AsyncCommandBufferCmdEmitterTest, RetainsBufferHazardsAcrossStreams) {
  ThunkSequence thunks;
  ThunkSequence body;
  Kernel(body, "async", 0);
  AsyncStartThunk* start =
      Start(thunks, ComputationStreamId(0), std::move(body));
  Kernel(thunks, "main", 0);
  Done(thunks, start, "done");

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options_));
  ASSERT_EQ(commands.size(), 3);
  EXPECT_TRUE(HappensBefore(commands, "async", "main"));
  EXPECT_TRUE(HappensBefore(commands, "main", "done"));
}

TEST_F(AsyncCommandBufferCmdEmitterTest,
       NestedRegionOnCurrentStreamStaysOrdered) {
  ThunkSequence thunks;
  ThunkSequence outer;
  ThunkSequence inner;
  Kernel(inner, "inner", 0);
  AsyncStartThunk* inner_start =
      Start(outer, ComputationStreamId(0), std::move(inner));
  Kernel(outer, "outer", 1);
  Done(outer, inner_start, "inner_done");
  AsyncStartThunk* outer_start =
      Start(thunks, ComputationStreamId(0), std::move(outer));
  Done(thunks, outer_start, "outer_done");

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options_));
  ASSERT_EQ(commands.size(), 4);
  EXPECT_TRUE(HappensBefore(commands, "inner", "outer"));
  EXPECT_TRUE(HappensBefore(commands, "outer", "inner_done"));
  EXPECT_TRUE(HappensBefore(commands, "inner_done", "outer_done"));
}

TEST_F(AsyncCommandBufferCmdEmitterTest, KeepsOrdinaryCommandsOrdered) {
  ThunkSequence thunks;
  Kernel(thunks, "first", 0);
  ThunkSequence nested;
  Kernel(nested, "second", 1);
  thunks.Emplace<SequentialThunk>(NextThunkInfo("sequential"),
                                  std::move(nested));
  Kernel(thunks, "third", 2);

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, options_));
  ASSERT_EQ(commands.size(), 3);
  EXPECT_TRUE(commands.execution_graph()->is_sequential());
  EXPECT_TRUE(HappensBefore(commands, "first", "second"));
  EXPECT_TRUE(HappensBefore(commands, "second", "third"));
}

TEST_F(AsyncCommandBufferCmdEmitterTest,
       OrdersSharedStreamsAcrossNestedExecutors) {
  for (bool use_while : {false, true}) {
    SCOPED_TRACE(use_while);
    BufferAllocation predicate(1, sizeof(int32_t), 0);
    BufferAllocation::Slice predicate_slice(&predicate, 0, sizeof(int32_t));
    ThunkSequence thunks;
    ThunkSequence first;
    Kernel(first, "first", 0);
    AsyncStartThunk* first_start =
        Start(thunks, ComputationStreamId(0), std::move(first));
    ThunkSequence independent;
    Kernel(independent, "independent", 1);
    AsyncStartThunk* independent_start =
        Start(thunks, CommunicationStreamId(0), std::move(independent));

    ThunkSequence nested;
    ThunkSequence body;
    Kernel(body, "nested", 2);
    AsyncStartThunk* nested_start =
        Start(nested, ComputationStreamId(0), std::move(body));
    Done(nested, nested_start, "nested_done");
    if (use_while) {
      thunks.Emplace<WhileThunk>(NextThunkInfo("while"), predicate_slice,
                                 ThunkSequence{}, std::move(nested));
    } else {
      std::vector<ThunkSequence> branches;
      branches.push_back(std::move(nested));
      thunks.Emplace<ConditionalThunk>(
          NextThunkInfo("conditional"),
          ShapedSlice{predicate_slice, ShapeUtil::MakeShape(S32, {})},
          std::move(branches));
    }
    Done(thunks, first_start, "first_done");
    Done(thunks, independent_start, "independent_done");

    ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                         ConvertToCommands(thunks, options_));
    ASSERT_EQ(commands.size(), 5);
    // Top-level nodes: first, independent, nested executor, two joins.
    // The nested executor must wait for earlier work on its computation stream,
    // while the unrelated communication stream can continue concurrently.
    auto graph = commands.execution_graph();
    EXPECT_THAT(graph->nodes_defs()[2].in_edges, ElementsAre(HasEdgeTo(0)));
    EXPECT_THAT(graph->nodes_defs()[3].in_edges, ElementsAre(HasEdgeTo(2)));
    EXPECT_THAT(graph->nodes_defs()[4].in_edges,
                UnorderedElementsAre(HasEdgeTo(1), HasEdgeTo(3)));
  }
}

TEST_F(AsyncCommandBufferCmdEmitterTest,
       OrdersSharedStreamsAcrossDynamicSliceExecutors) {
  for (Thunk::Kind wrapper :
       {Thunk::kDynamicSliceFusion, Thunk::kWhile, Thunk::kConditional}) {
    SCOPED_TRACE(Thunk::KindToString(wrapper));
    BufferAllocation result_allocation(1, 1024, 0);
    BufferAllocation predicate(2, sizeof(int32_t), 0);
    ThunkSequence thunks;
    ThunkSequence first;
    Kernel(first, "first", 1);
    AsyncStartThunk* first_start =
        Start(thunks, ComputationStreamId(0), std::move(first));
    ThunkSequence independent;
    Kernel(independent, "independent", 0);
    AsyncStartThunk* independent_start =
        Start(thunks, CommunicationStreamId(0), std::move(independent));

    // The private allocation deliberately has the same index and offset as
    // "independent". It must not become a buffer dependency in the outer graph.
    std::vector<BufferAllocation> embedded_allocations;
    embedded_allocations.emplace_back(0, 1024, 0);
    BufferAllocation::Slice embedded_slice(&embedded_allocations[0], 0, 1024);
    ThunkSequence embedded_body;
    embedded_body.Emplace<FakeKernelThunk>(NextThunkInfo("embedded"),
                                           embedded_slice);
    ThunkSequence embedded;
    AsyncStartThunk* embedded_start =
        Start(embedded, ComputationStreamId(0), std::move(embedded_body));
    Done(embedded, embedded_start, "embedded_done");

    Shape shape = ShapeUtil::MakeShape(F32, {256});
    ThunkSequence nested;
    nested.Emplace<DynamicSliceFusionV2Thunk>(
        NextThunkInfo("dynamic_slice"),
        std::vector<DynamicSliceFusion::Parameter>{},
        std::vector<DynamicSliceFusion::Result>{
            {std::nullopt, 0, shape, shape}},
        std::vector<BufferAllocation::Slice>{},
        std::vector<BufferAllocation::Slice>{
            BufferAllocation::Slice(&result_allocation, 0, 1024)},
        std::move(embedded_allocations), std::move(embedded));

    // Also discover the hidden stream when the fusion is inside control flow.
    if (wrapper == Thunk::kWhile) {
      ThunkSequence body = std::move(nested);
      nested.Emplace<WhileThunk>(
          NextThunkInfo("while"), BufferAllocation::Slice(&predicate, 0, 1),
          ThunkSequence{}, std::move(body), /*trip_count=*/1);
    } else if (wrapper == Thunk::kConditional) {
      std::vector<ThunkSequence> branches;
      branches.push_back(std::move(nested));
      nested.Emplace<ConditionalThunk>(
          NextThunkInfo("conditional"),
          ShapedSlice{BufferAllocation::Slice(&predicate, 0, sizeof(int32_t)),
                      ShapeUtil::MakeShape(S32, {})},
          std::move(branches));
    }

    // Running the outer executor on a different async stream ensures that
    // "last" cannot accidentally inherit its completion from the main stream.
    AsyncStartThunk* outer_start =
        Start(thunks, ComputationStreamId(1), std::move(nested));
    ThunkSequence last;
    Kernel(last, "last", 3);
    AsyncStartThunk* last_start =
        Start(thunks, ComputationStreamId(0), std::move(last));
    Done(thunks, first_start, "first_done");
    Done(thunks, outer_start, "outer_done");
    Done(thunks, last_start, "last_done");
    Done(thunks, independent_start, "independent_done");

    ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                         ConvertToCommands(thunks, options_));
    ASSERT_EQ(commands.size(), 8);
    // Top-level nodes: first, independent, nested executor, last, four joins.
    // Same-stream ordering requires first -> nested executor -> last, while
    // the unrelated communication stream only joins at its own done.
    auto graph = commands.execution_graph();
    EXPECT_THAT(graph->nodes_defs()[2].in_edges, ElementsAre(HasEdgeTo(0)));
    EXPECT_THAT(graph->nodes_defs()[3].in_edges, ElementsAre(HasEdgeTo(2)));
    EXPECT_THAT(graph->nodes_defs()[1].out_edges, ElementsAre(HasEdgeTo(7)));
  }
}

TEST_F(AsyncCommandBufferCmdEmitterTest, RejectsOverlappingSharedExecution) {
  for (Thunk::Kind wrapper :
       {Thunk::kSequential, Thunk::kAsyncStart, Thunk::kWhile,
        Thunk::kConditional, Thunk::kDynamicSliceFusion}) {
    SCOPED_TRACE(Thunk::KindToString(wrapper));
    auto canonical = std::make_unique<AsyncStartThunk>(
        NextThunkInfo("canonical"), ComputationStreamId(0), ThunkSequence{});
    ThunkSequence inner;
    inner.Emplace<AsyncStartThunk>(NextThunkInfo("second"),
                                   ComputationStreamId(0), ThunkSequence{},
                                   canonical->async_execution());
    Done(inner, canonical.get(), "second_done");

    BufferAllocation predicate(1, sizeof(int32_t), 0);
    ThunkSequence body;
    if (wrapper == Thunk::kAsyncStart) {
      body = std::move(inner);
    } else if (wrapper == Thunk::kWhile) {
      body.Emplace<WhileThunk>(
          NextThunkInfo("while"), BufferAllocation::Slice(&predicate, 0, 1),
          ThunkSequence{}, std::move(inner), /*trip_count=*/1);
    } else if (wrapper == Thunk::kConditional) {
      std::vector<ThunkSequence> branches;
      branches.push_back(std::move(inner));
      body.Emplace<ConditionalThunk>(
          NextThunkInfo("conditional"),
          ShapedSlice{BufferAllocation::Slice(&predicate, 0, sizeof(int32_t)),
                      ShapeUtil::MakeShape(S32, {})},
          std::move(branches));
    } else if (wrapper == Thunk::kDynamicSliceFusion) {
      body.Emplace<DynamicSliceFusionV2Thunk>(
          NextThunkInfo("dynamic_slice"),
          std::vector<DynamicSliceFusion::Parameter>{},
          std::vector<DynamicSliceFusion::Result>{},
          std::vector<BufferAllocation::Slice>{},
          std::vector<BufferAllocation::Slice>{},
          std::vector<BufferAllocation>{}, std::move(inner));
    }

    ThunkSequence thunks;
    thunks.Emplace<AsyncStartThunk>(NextThunkInfo("first"),
                                    ComputationStreamId(1), std::move(body),
                                    canonical->async_execution());
    if (wrapper == Thunk::kSequential) {
      thunks.Emplace<SequentialThunk>(NextThunkInfo("sequential"),
                                      std::move(inner));
    }
    Done(thunks, canonical.get(), "first_done");

    EXPECT_THAT(ConvertToCommands(thunks, options_),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("Async execution already started")));
  }
}

TEST_F(AsyncCommandBufferCmdEmitterTest, RejectsUnmatchedAsyncBoundaries) {
  ThunkSequence start_only;
  AsyncStartThunk* start = Start(start_only, ComputationStreamId(0), {});
  EXPECT_THAT(ConvertToCommands(start_only, options_),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Async start has no matching done")));
  ThunkSequence done_only;
  Done(done_only, start, "done");
  EXPECT_THAT(ConvertToCommands(done_only, options_),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Async done has no matching start")));
}

}  // namespace
}  // namespace xla::gpu
