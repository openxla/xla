/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/rpc/client/save_profile.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"  // IWYU pragma: keep
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;

TEST(SaveProfileTest, SaveXSpaceChunk) {
  std::string temp_dir = ::testing::TempDir();
  std::string run = "test_run";
  std::string host = "test_host";

  XSpace space1;
  space1.add_hostnames("host1");
  XPlane* plane1 = space1.add_planes();
  plane1->set_name("plane1");

  // Save chunk 0 (creates new file)
  ASSERT_OK(SaveXSpaceChunk(temp_dir, run, host, 0, space1));

  std::string file_path =
      io::JoinPath(temp_dir, run, "test_host.xplane.riegeli");
  EXPECT_TRUE(Env::Default()->FileExists(file_path).ok());

  // Save chunk 1 (appends)
  XSpace space2;
  space2.add_hostnames("host2");
  XPlane* plane2 = space2.add_planes();
  plane2->set_name("plane2");
  ASSERT_OK(SaveXSpaceChunk(temp_dir, run, host, 1, space2));

  // Check file size is multiple of 64KB
  uint64_t file_size = 0;
  ASSERT_OK(Env::Default()->GetFileSize(file_path, &file_size));
  EXPECT_EQ(file_size % (64 * 1024), 0);

  // Read records back
  std::vector<XSpace> read_spaces;
  riegeli::RecordReader<riegeli::FdReader<>> reader{
      riegeli::FdReader<>(file_path)};
  XSpace read_space;
  while (reader.ReadRecord(read_space)) {
    read_spaces.push_back(read_space);
    read_space.Clear();
  }
  ASSERT_OK(reader.status());
  reader.Close();

  ASSERT_EQ(read_spaces.size(), 2);
  EXPECT_EQ(read_spaces[0].hostnames(0), "host1");
  EXPECT_EQ(read_spaces[0].planes(0).name(), "plane1");
  EXPECT_EQ(read_spaces[1].hostnames(0), "host2");
  EXPECT_EQ(read_spaces[1].planes(0).name(), "plane2");

  // Clean up
  ASSERT_OK(Env::Default()->DeleteFile(file_path));
}

struct DummyOptionsNew {
  bool set_padding_called = false;
  void set_padding(size_t size) {
    if (size == 64 * 1024) {
      set_padding_called = true;
    }
  }
};

struct DummyOptionsOld {
  bool set_pad_to_block_boundary_called = false;
  void set_pad_to_block_boundary(bool pad) {
    if (pad) {
      set_pad_to_block_boundary_called = true;
    }
  }
};

template <typename T, typename = void>
struct TestHasSetPadding : std::false_type {};

template <typename T>
struct TestHasSetPadding<
    T, std::void_t<decltype(std::declval<T>().set_padding(64 * 1024))>>
    : std::true_type {};

template <typename T, typename = void>
struct TestHasSetPadToBlockBoundary : std::false_type {};

template <typename T>
struct TestHasSetPadToBlockBoundary<
    T, std::void_t<decltype(std::declval<T>().set_pad_to_block_boundary(true))>>
    : std::true_type {};

template <typename Options>
void TestSetPadding(Options& options) {
  if constexpr (TestHasSetPadding<Options>::value) {
    options.set_padding(64 * 1024);
  } else if constexpr (TestHasSetPadToBlockBoundary<Options>::value) {
    options.set_pad_to_block_boundary(true);
  }
}

TEST(SaveProfileTest, SetPaddingCompatibility) {
  DummyOptionsNew new_opts;
  TestSetPadding(new_opts);
  EXPECT_TRUE(new_opts.set_padding_called);

  DummyOptionsOld old_opts;
  TestSetPadding(old_opts);
  EXPECT_TRUE(old_opts.set_pad_to_block_boundary_called);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
