/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/lib/io/random_inputstream.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace io {
namespace {

TEST(RandomInputStream, ReadNBytes) {
  Env* env = Env::Default();
  std::string fname = testing::TmpDir() + "/random_inputbuffer_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));

  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));
  tstring read;
  RandomAccessInputStream in(file.get());
  TF_ASSERT_OK(in.ReadNBytes(3, &read));
  EXPECT_EQ(read, "012");
  EXPECT_EQ(3, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "");
  EXPECT_EQ(3, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(5, &read));
  EXPECT_EQ(read, "34567");
  EXPECT_EQ(8, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "");
  EXPECT_EQ(8, in.Tell());
  EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(20, &read)));
  EXPECT_EQ(read, "89");
  EXPECT_EQ(10, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "");
  EXPECT_EQ(10, in.Tell());
}

#if defined(TF_CORD_SUPPORT)
TEST(RandomInputStream, ReadNBytesWithCords) {
  Env* env = Env::Default();
  std::string fname = testing::TmpDir() + "/random_inputbuffer_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));

  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));
  absl::Cord read;
  RandomAccessInputStream in(file.get());

  // Reading into `absl::Cord`s does not clear existing data from the cord.
  TF_ASSERT_OK(in.ReadNBytes(3, &read));
  EXPECT_EQ(read, "012");
  EXPECT_EQ(3, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "012");
  EXPECT_EQ(3, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(5, &read));
  EXPECT_EQ(read, "01234567");
  EXPECT_EQ(8, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "01234567");
  EXPECT_EQ(8, in.Tell());
  EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(20, &read)));
  EXPECT_EQ(read, "0123456789");
  EXPECT_EQ(10, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "0123456789");
  EXPECT_EQ(10, in.Tell());
}
#endif

TEST(RandomInputStream, SkipNBytes) {
  Env* env = Env::Default();
  std::string fname = testing::TmpDir() + "/random_inputbuffer_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));

  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));
  tstring read;
  RandomAccessInputStream in(file.get());
  TF_ASSERT_OK(in.SkipNBytes(3));
  EXPECT_EQ(3, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(0, &read));
  EXPECT_EQ(read, "");
  EXPECT_EQ(3, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(4, &read));
  EXPECT_EQ(read, "3456");
  EXPECT_EQ(7, in.Tell());
  TF_ASSERT_OK(in.SkipNBytes(0));
  EXPECT_EQ(7, in.Tell());
  TF_ASSERT_OK(in.ReadNBytes(2, &read));
  EXPECT_EQ(read, "78");
  EXPECT_EQ(9, in.Tell());
  EXPECT_TRUE(absl::IsOutOfRange(in.SkipNBytes(20)));
  EXPECT_EQ(10, in.Tell());
  // Making sure that if we read after we've skipped beyond end of file, we get
  // nothing.
  EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(5, &read)));
  EXPECT_EQ(read, "");
  EXPECT_EQ(10, in.Tell());
}

TEST(RandomInputStream, Seek) {
  Env* env = Env::Default();
  std::string fname = testing::TmpDir() + "/random_inputbuffer_seek_test";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));

  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));
  tstring read;
  RandomAccessInputStream in(file.get());

  // Seek forward
  TF_ASSERT_OK(in.Seek(3));
  EXPECT_EQ(3, in.Tell());

  // Read 4 bytes
  TF_ASSERT_OK(in.ReadNBytes(4, &read));
  EXPECT_EQ(read, "3456");
  EXPECT_EQ(7, in.Tell());

  // Seek backwards
  TF_ASSERT_OK(in.Seek(1));
  TF_ASSERT_OK(in.ReadNBytes(4, &read));
  EXPECT_EQ(read, "1234");
  EXPECT_EQ(5, in.Tell());
}

TEST(RandomInputStream, ZeroByteAndNullEofSliceReadSafe) {
  constexpr int kOutOfRangeCode = 11;
  Env* env = Env::Default();
  const std::string empty_ram = "ram://random_inputstream_empty.txt";
  const std::string small_ram = "ram://random_inputstream_small.txt";
  TF_ASSERT_OK(WriteStringToFile(env, empty_ram, ""));
  TF_ASSERT_OK(WriteStringToFile(env, small_ram, "0123"));

  // RamRandomAccessFile leaves the output string_view as `(nullptr, 0)` at EOF;
  // both 0-byte reads and non-zero reads at EOF must avoid nonnull memmove UB.
  std::unique_ptr<RandomAccessFile> empty_file;
  TF_ASSERT_OK(env->NewRandomAccessFile(empty_ram, &empty_file));
  RandomAccessInputStream empty_in(empty_file.get());
  tstring out = "stale";
  TF_ASSERT_OK(empty_in.ReadNBytes(0, &out));
  EXPECT_EQ(out, "");
  EXPECT_EQ(empty_in.Tell(), 0);
  EXPECT_EQ(static_cast<int>(empty_in.ReadNBytes(8, &out).code()),
            kOutOfRangeCode);
  EXPECT_EQ(out, "");

  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(small_ram, &file));
  RandomAccessInputStream in(file.get());
  EXPECT_EQ(static_cast<int>(in.ReadNBytes(8, &out).code()), kOutOfRangeCode);
  EXPECT_EQ(out, "0123");
  EXPECT_EQ(in.Tell(), 4);
  EXPECT_EQ(static_cast<int>(in.ReadNBytes(8, &out).code()), kOutOfRangeCode);
  EXPECT_EQ(out, "");
  EXPECT_EQ(in.Tell(), 4);
  TF_ASSERT_OK(in.ReadNBytes(0, &out));
  EXPECT_EQ(out, "");

#if defined(TF_CORD_SUPPORT)
  absl::Cord cord("keep");
  TF_ASSERT_OK(in.ReadNBytes(0, &cord));
  EXPECT_EQ(cord, "keep");
  EXPECT_EQ(in.Tell(), 4);
#endif
  TF_ASSERT_OK(env->DeleteFile(empty_ram));
  TF_ASSERT_OK(env->DeleteFile(small_ram));
}

#if defined(TF_CORD_SUPPORT)
class MockCordRandomAccessFile : public RandomAccessFile {
 public:
  using StatusType = decltype(RandomAccessInputStream(nullptr).Reset());
  explicit MockCordRandomAccessFile(bool shrink, size_t append_bytes)
      : shrink_(shrink), append_bytes_(append_bytes) {}

  StatusType Read(uint64_t offset, size_t n, absl::Cord* cord) const override {
    if (shrink_) {
      cord->Clear();
    } else {
      cord->Append(std::string(append_bytes_, 'x'));
    }
    return StatusType();
  }

 private:
  bool shrink_;
  size_t append_bytes_;
};
#endif

TEST(RandomInputStream, RejectsNullNegativeSeekAndInvalidCord) {
  constexpr int kInvalidArgumentCode = 3;
  RandomAccessInputStream null_in(nullptr);
  tstring out = "stale";
  EXPECT_EQ(null_in.Tell(), -1);
  EXPECT_EQ(static_cast<int>(null_in.ReadNBytes(4, &out).code()),
            kInvalidArgumentCode);
  EXPECT_EQ(out, "");
  EXPECT_EQ(static_cast<int>(null_in.ReadNBytes(0, &out).code()),
            kInvalidArgumentCode);
  EXPECT_EQ(static_cast<int>(
                null_in.ReadNBytes(4, static_cast<tstring*>(nullptr)).code()),
            kInvalidArgumentCode);
  EXPECT_EQ(static_cast<int>(null_in.SkipNBytes(0).code()),
            kInvalidArgumentCode);
  EXPECT_EQ(static_cast<int>(null_in.SkipNBytes(4).code()),
            kInvalidArgumentCode);

  Env* env = Env::Default();
  const std::string fname = testing::TmpDir() + "/random_inputstream_neg_seek";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));
  RandomAccessInputStream in(file.get());
  TF_ASSERT_OK(in.Seek(-1));
  EXPECT_EQ(static_cast<int>(in.ReadNBytes(1, &out).code()),
            kInvalidArgumentCode);
  EXPECT_EQ(static_cast<int>(in.ReadNBytes(0, &out).code()),
            kInvalidArgumentCode);
  EXPECT_EQ(static_cast<int>(in.SkipNBytes(1).code()), kInvalidArgumentCode);
  EXPECT_EQ(static_cast<int>(in.SkipNBytes(0).code()), kInvalidArgumentCode);
#if defined(TF_CORD_SUPPORT)
  constexpr int kOutOfRangeCode = 11;
  constexpr int kInternalCode = 13;
  absl::Cord cord("prefix");
  EXPECT_EQ(
      static_cast<int>(
          null_in.ReadNBytes(4, static_cast<absl::Cord*>(nullptr)).code()),
      kInvalidArgumentCode);
  EXPECT_EQ(static_cast<int>(null_in.ReadNBytes(4, &cord).code()),
            kInvalidArgumentCode);
  EXPECT_EQ(static_cast<int>(in.ReadNBytes(1, &cord).code()),
            kInvalidArgumentCode);
  EXPECT_EQ(static_cast<int>(in.ReadNBytes(0, &cord).code()),
            kInvalidArgumentCode);
  EXPECT_EQ(cord, "prefix");

  MockCordRandomAccessFile shrink_file(/*shrink=*/true, 0);
  RandomAccessInputStream shrink_in(&shrink_file);
  TF_ASSERT_OK(shrink_in.Seek(5));
  EXPECT_EQ(static_cast<int>(shrink_in.ReadNBytes(3, &cord).code()),
            kInternalCode);
  EXPECT_EQ(shrink_in.Tell(), 5);

  MockCordRandomAccessFile grow_file(/*shrink=*/false, 8);
  RandomAccessInputStream grow_in(&grow_file);
  absl::Cord grown;
  EXPECT_EQ(static_cast<int>(grow_in.ReadNBytes(3, &grown).code()),
            kInternalCode);
  EXPECT_EQ(grow_in.Tell(), 0);

  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  MockCordRandomAccessFile overflow_file(/*shrink=*/false, 5);
  RandomAccessInputStream overflow_in(&overflow_file);
  TF_ASSERT_OK(overflow_in.Seek(kMax - 2));
  absl::Cord capped("p");
  EXPECT_EQ(static_cast<int>(overflow_in.ReadNBytes(5, &capped).code()),
            kOutOfRangeCode);
  EXPECT_EQ(capped, "pxx");
  EXPECT_EQ(overflow_in.Tell(), kMax);
#endif
  TF_ASSERT_OK(in.Reset());
  TF_ASSERT_OK(in.ReadNBytes(3, &out));
  EXPECT_EQ(out, "012");
}

TEST(RandomInputStream, ReadNBytesInt64MaxAndMultiChunk) {
  constexpr int kOutOfRangeCode = 11;
  Env* env = Env::Default();
  const std::string small_name =
      testing::TmpDir() + "/random_inputstream_int64max";
  const std::string expected(256, 'a');
  TF_ASSERT_OK(WriteStringToFile(env, small_name, expected));
  std::unique_ptr<RandomAccessFile> small_file;
  TF_ASSERT_OK(env->NewRandomAccessFile(small_name, &small_file));

  RandomAccessInputStream small_in(small_file.get());
  tstring out;
  EXPECT_EQ(static_cast<int>(
                small_in.ReadNBytes(std::numeric_limits<int64_t>::max(), &out)
                    .code()),
            kOutOfRangeCode);
  EXPECT_EQ(out, expected);
  EXPECT_LT(out.capacity(), 4096);
  EXPECT_EQ(small_in.Tell(), 256);

  constexpr size_t kLargeSize = 17 * 1024 * 1024 + 19;
  const std::string large_name = "ram://random_inputstream_multichunk.bin";
  std::string large_data(kLargeSize, '\0');
  for (size_t i = 0; i < kLargeSize; ++i) {
    large_data[i] = static_cast<char>((i * 131 + 17) & 0xFF);
  }
  TF_ASSERT_OK(WriteStringToFile(env, large_name, large_data));
  std::unique_ptr<RandomAccessFile> large_file;
  TF_ASSERT_OK(env->NewRandomAccessFile(large_name, &large_file));

  {
    RandomAccessInputStream in(large_file.get());
    TF_ASSERT_OK(in.ReadNBytes(kLargeSize, &out));
    ASSERT_EQ(out.size(), kLargeSize);
    EXPECT_EQ(out, large_data);
    EXPECT_EQ(in.Tell(), static_cast<int64_t>(kLargeSize));
  }
  {
    RandomAccessInputStream in(large_file.get());
    EXPECT_EQ(static_cast<int>(in.ReadNBytes(25 * 1024 * 1024, &out).code()),
              kOutOfRangeCode);
    EXPECT_EQ(out.size(), kLargeSize);
    EXPECT_EQ(in.Tell(), static_cast<int64_t>(kLargeSize));
  }
  TF_ASSERT_OK(env->DeleteFile(large_name));
}

TEST(RandomInputStream, SkipNBytesInt64MaxNoSignedOverflow) {
  constexpr int kOutOfRangeCode = 11;
  Env* env = Env::Default();
  const std::string fname =
      testing::TmpDir() + "/random_inputstream_skip_int64max";
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  for (int64_t start : {int64_t{0}, int64_t{1}, int64_t{9}, int64_t{10}}) {
    RandomAccessInputStream in(file.get());
    TF_ASSERT_OK(in.Seek(start));
    EXPECT_EQ(static_cast<int>(in.SkipNBytes(kMax).code()), kOutOfRangeCode);
    EXPECT_EQ(in.Tell(), 10);
  }
}

}  // anonymous namespace
}  // namespace io
}  // namespace tsl
