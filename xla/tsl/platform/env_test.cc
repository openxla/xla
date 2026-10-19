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

#include "xla/tsl/platform/env.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tsl/platform/null_file_system.h"
#include "tsl/platform/path.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "xla/tsl/testing/temporary_directory.h"

namespace tsl {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(EnvTest, StartDetachedThread) {
  Env* env = Env::Default();
  const int num_threads = 10;
  absl::BlockingCounter counter(num_threads);

  ThreadOptions thread_options;
  for (int i = 0; i < num_threads; ++i) {
    env->StartDetachedThread(thread_options, "MyDetachedThread", [&]() {
      absl::SleepFor(absl::Milliseconds(50));
      counter.DecrementCount();
    });
  }

  counter.Wait();
}

TEST(EnvTest, FileOperations) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string file_path = tsl::io::JoinPath(temp_dir.path(), "test.txt");
  EXPECT_THAT(WriteStringToFile(env, file_path, "test1"), IsOk());
  EXPECT_THAT(AppendStringToFile(env, file_path, "test2"), IsOk());
  std::string content;
  EXPECT_THAT(ReadFileToString(env, file_path, &content), IsOk());
  EXPECT_EQ(content, "test1test2");
}

TEST(EnvTest, ReadFileToStringEmptyFile) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string file_path = tsl::io::JoinPath(temp_dir.path(), "empty.txt");
  EXPECT_THAT(WriteStringToFile(env, file_path, ""), IsOk());
  std::string content = "pre_populated_garbage";
  EXPECT_THAT(ReadFileToString(env, file_path, &content), IsOk());
  EXPECT_TRUE(content.empty());
}

TEST(EnvTest, SimpleFileSystemConformance) {
  std::vector<std::string> schemes;
  Env* env = Env::Default();
  ASSERT_OK(env->GetRegisteredFileSystemSchemes(&schemes));

  ASSERT_OK_AND_ASSIGN(testing::TemporaryDirectory temp_dir,
                       testing::TemporaryDirectory::CreateForCurrentTestcase());
  for (const absl::string_view scheme : schemes) {
    std::string contents = "data";
    std::string file_path;
    if (!scheme.empty()) {
      file_path = io::JoinPath(
          absl::StrCat(scheme, "://", temp_dir.path(), "data.txt"));
    } else {
      file_path = io::JoinPath(temp_dir.path(), "data.txt");
    }
    EXPECT_OK(WriteStringToFile(env, file_path, contents));
    EXPECT_OK(WriteStringToFile(env, file_path, contents));
    std::string content;
    EXPECT_OK(ReadFileToString(env, file_path, &content));
    EXPECT_EQ(content, contents) << file_path;
  }
}

TEST(EnvTest, RenameFile) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string src_path = tsl::io::JoinPath(temp_dir.path(), "src.txt");
  std::string target_path = tsl::io::JoinPath(temp_dir.path(), "target.txt");
  EXPECT_THAT(WriteStringToFile(env, src_path, "source content"), IsOk());
  EXPECT_THAT(env->FileExists(src_path), IsOk());

  EXPECT_THAT(WriteStringToFile(env, target_path, "target content"), IsOk());
  EXPECT_THAT(env->RenameFile(src_path, target_path, /*overwrite=*/true),
              IsOk());
  EXPECT_TRUE(absl::IsNotFound(env->FileExists(src_path)));
  std::string target_content;
  EXPECT_THAT(ReadFileToString(env, target_path, &target_content), IsOk());
  EXPECT_EQ(target_content, "source content");
}

TEST(EnvTest, GetMatchingPaths) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string dir_a = tsl::io::JoinPath(temp_dir.path(), "dir_a");
  std::string dir_b = tsl::io::JoinPath(temp_dir.path(), "dir_b");
  ASSERT_THAT(env->RecursivelyCreateDir(dir_a), IsOk());
  ASSERT_THAT(env->RecursivelyCreateDir(dir_b), IsOk());

  std::string file_1 = tsl::io::JoinPath(dir_a, "file_1.txt");
  std::string file_2 = tsl::io::JoinPath(dir_b, "file_2.txt");
  ASSERT_THAT(WriteStringToFile(env, file_1, "1"), IsOk());
  ASSERT_THAT(WriteStringToFile(env, file_2, "2"), IsOk());

  std::string pattern =
      tsl::io::JoinPath(temp_dir.path(), "dir_*", "file_*.txt");
  absl::string_view pattern_sv = pattern;
  for (bool multithreaded : {false, true}) {
    std::vector<std::string> results;
    EXPECT_THAT(env->GetMatchingPaths(pattern, &results, multithreaded),
                IsOk());
    EXPECT_THAT(results, ElementsAre(file_1, file_2));

    results.clear();
    EXPECT_THAT(env->GetMatchingPaths(pattern_sv, &results, multithreaded),
                IsOk());
    EXPECT_THAT(results, ElementsAre(file_1, file_2));
  }
}

// Returns the matches in reverse lexicographic order.
class ReverseOrderFileSystem : public NullFileSystem {
 public:
  absl::Status GetMatchingPaths(const std::string& pattern,
                                std::vector<std::string>* results) override {
    *results = {"c", "b", "a"};
    return absl::OkStatus();
  }
};

// Uses `fs` for all paths.
class SingleFileSystemEnv : public EnvWrapper {
 public:
  explicit SingleFileSystemEnv(FileSystem* fs)
      : EnvWrapper(Env::Default()), fs_(fs) {}

  absl::Status GetFileSystemForFile(absl::string_view fname,
                                    FileSystem** result) override {
    *result = fs_;
    return absl::OkStatus();
  }

 private:
  FileSystem* fs_;
};

TEST(EnvTest, GetMatchingPathsSortsResultsOfAnyFileSystem) {
  ReverseOrderFileSystem fs;
  SingleFileSystemEnv env(&fs);

  std::vector<std::string> results;
  ASSERT_OK(env.GetMatchingPaths("*", &results));
  EXPECT_THAT(results, ElementsAre("a", "b", "c"));
}

#if defined(__linux__)
TEST(EnvTest, ReadFileToStringAllowsShortReadsForSystemFiles) {
  std::string content;
  absl::Status s = ReadFileToString(
      Env::Default(), "/sys/devices/system/cpu/kernel_max", &content);
  if (absl::IsNotFound(s)) {
    GTEST_SKIP() << "/sys/devices/system/cpu/kernel_max not available: " << s;
  }
  EXPECT_OK(s);
  EXPECT_FALSE(content.empty());
}
#endif

class FailingReadRandomAccessFile : public RandomAccessFile {
 public:
  FailingReadRandomAccessFile(std::string prefix, absl::Status status,
                              bool fail_on_first_call = true)
      : prefix_(std::move(prefix)),
        status_(std::move(status)),
        fail_on_first_call_(fail_on_first_call) {}

  absl::Status Read(uint64_t offset, absl::string_view& result,
                    absl::Span<char> scratch) const override {
    if (offset < prefix_.size()) {
      const size_t n = std::min(scratch.size(), prefix_.size() - offset);
      std::memcpy(scratch.data(), prefix_.data() + offset, n);
      result = absl::string_view(scratch.data(), n);
      return fail_on_first_call_ ? status_ : absl::OkStatus();
    }
    result = absl::string_view();
    return status_;
  }

 private:
  std::string prefix_;
  absl::Status status_;
  bool fail_on_first_call_;
};

class FailingRandomAccessFileSystem : public NullFileSystem {
 public:
  FailingRandomAccessFileSystem(std::string prefix, absl::Status status,
                                bool fail_on_first_call = true)
      : prefix_(std::move(prefix)),
        status_(std::move(status)),
        fail_on_first_call_(fail_on_first_call) {}

  absl::Status NewRandomAccessFile(
      const std::string& fname,
      std::unique_ptr<RandomAccessFile>* result) override {
    *result = std::make_unique<FailingReadRandomAccessFile>(
        prefix_, status_, fail_on_first_call_);
    return absl::OkStatus();
  }

 private:
  std::string prefix_;
  absl::Status status_;
  bool fail_on_first_call_;
};

TEST(EnvTest, ReadProtoEmptyAndCorruptFiles) {
  Env* env = Env::Default();
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());

  const std::string empty_path =
      tsl::io::JoinPath(temp_dir.path(), "empty.proto");
  ASSERT_THAT(WriteStringToFile(env, empty_path, ""), IsOk());
  tensorflow::StatusProto proto;
  EXPECT_THAT(ReadBinaryProto(env, empty_path, &proto), IsOk());
  EXPECT_THAT(ReadTextProto(env, empty_path, &proto), IsOk());

  const std::string corrupt_path =
      tsl::io::JoinPath(temp_dir.path(), "corrupt.proto");
  ASSERT_THAT(WriteStringToFile(env, corrupt_path, "\xff\xff\xff"), IsOk());
  EXPECT_THAT(ReadBinaryProto(env, corrupt_path, &proto),
              StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(ReadTextProto(env, corrupt_path, &proto),
              StatusIs(absl::StatusCode::kDataLoss));
}

TEST(EnvTest, ReadProtoPropagatesLazyRandomAccessFileError) {
  FailingRandomAccessFileSystem lazy_fs(
      /*prefix=*/"", absl::NotFoundError("Object not found"));
  SingleFileSystemEnv env(&lazy_fs);

  tensorflow::StatusProto proto;
  EXPECT_THAT(ReadBinaryProto(&env, "missing.pb", &proto),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(ReadTextProto(&env, "missing.txtpb", &proto),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(ReadTextOrBinaryProto(&env, "missing", &proto),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(EnvTest, ReadProtoPropagatesPartialReadError) {
  tensorflow::StatusProto valid_proto;
  valid_proto.set_code(tensorflow::error::INVALID_ARGUMENT);
  valid_proto.set_message("partial prefix");
  const std::string valid_binary = valid_proto.SerializeAsString();
  const std::string valid_text = "code: INVALID_ARGUMENT\n";

  for (bool fail_on_first_call : {true, false}) {
    SCOPED_TRACE(fail_on_first_call);
    FailingRandomAccessFileSystem bin_fs(valid_binary,
                                         absl::UnavailableError("Stream reset"),
                                         fail_on_first_call);
    SingleFileSystemEnv bin_env(&bin_fs);
    tensorflow::StatusProto proto;
    EXPECT_THAT(ReadBinaryProto(&bin_env, "truncated.pb", &proto),
                StatusIs(absl::StatusCode::kUnavailable));

    FailingRandomAccessFileSystem txt_fs(
        valid_text, absl::UnavailableError("Stream reset"), fail_on_first_call);
    SingleFileSystemEnv txt_env(&txt_fs);
    EXPECT_THAT(ReadTextProto(&txt_env, "truncated.txtpb", &proto),
                StatusIs(absl::StatusCode::kUnavailable));
  }
}

}  // namespace
}  // namespace tsl
