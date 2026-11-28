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

#include "xla/tsl/lib/io/buffered_inputstream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/io/random_inputstream.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace tsl {
namespace io {
namespace {

static std::vector<int> BufferSizes() {
  return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   11,
          12, 13, 14, 15, 16, 17, 18, 19, 20, 65536};
}

// This class will only return OutOfRange error once to make sure that
// BufferedInputStream is able to cache the error.
class ReadOnceInputStream : public InputStreamInterface {
 public:
  ReadOnceInputStream() : start_(true) {}

  absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) override {
    if (bytes_to_read < 11) {
      return absl::InvalidArgumentError(
          absl::StrCat("Not reading all bytes: ", bytes_to_read));
    }
    if (start_) {
      *result = "0123456789";
      start_ = false;
      return absl::OutOfRangeError("Out of range.");
    }
    return absl::InvalidArgumentError(
        "Redudant call to ReadNBytes after an OutOfRange error.");
  }

  int64_t Tell() const override { return start_ ? 0 : 10; }

  // Resets the stream to the beginning.
  absl::Status Reset() override {
    start_ = true;
    return absl::OkStatus();
  }

 private:
  bool start_;
};

TEST(BufferedInputStream, ReadLine_Empty) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, ""));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    std::string line;
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine1) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(
      WriteStringToFile(env, fname, "line one\nline two\nline three\n"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    std::string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine_NoTrailingNewLine) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "line one\nline two\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    std::string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine_EmptyLines) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(
      WriteStringToFile(env, fname, "line one\n\n\nline two\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    std::string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine_CRLF) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname,
                                 "line one\r\n\r\n\r\nline two\r\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    std::string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, SkipLine1) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(
      WriteStringToFile(env, fname, "line one\nline two\nline three\n"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    std::string line;
    TF_ASSERT_OK(in.SkipLine());
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.SkipLine());
    EXPECT_TRUE(absl::IsOutOfRange(in.SkipLine()));
    // A second call should also return end of file
    EXPECT_TRUE(absl::IsOutOfRange(in.SkipLine()));
  }
}

TEST(BufferedInputStream, SkipLine_NoTrailingNewLine) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "line one\nline two\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    std::string line;
    TF_ASSERT_OK(in.SkipLine());
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.SkipLine());
    EXPECT_TRUE(absl::IsOutOfRange(in.SkipLine()));
    // A second call should also return end of file
    EXPECT_TRUE(absl::IsOutOfRange(in.SkipLine()));
  }
}

TEST(BufferedInputStream, SkipLine_EmptyLines) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "line one\n\n\nline two"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    std::string line;
    TF_ASSERT_OK(in.SkipLine());
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.SkipLine());
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
  }
}

TEST(BufferedInputStream, ReadNBytes) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    tstring read;
    BufferedInputStream in(input_stream.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(7, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "789");
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, OutOfRangeCache) {
  for (auto buf_size : BufferSizes()) {
    if (buf_size < 11) {
      continue;
    }
    ReadOnceInputStream input_stream;
    tstring read;
    BufferedInputStream in(&input_stream, buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK((in.ReadNBytes(7, &read)));
    EXPECT_EQ(read, "3456789");
    EXPECT_EQ(10, in.Tell());
    absl::Status s = in.ReadNBytes(5, &read);
    // Make sure the read is failing with OUT_OF_RANGE error. If it is failing
    // with other errors, it is not caching the OUT_OF_RANGE properly.
    EXPECT_EQ(error::OUT_OF_RANGE, s.code()) << s;
    EXPECT_EQ(read, "");
    // Empty read shouldn't cause an error even at the end of the file.
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
  }
}

TEST(BufferedInputStream, SkipNBytes) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    tstring read;
    BufferedInputStream in(input_stream.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(3));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(2, &read));
    EXPECT_EQ(read, "34");
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(2));
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(1, &read));
    EXPECT_EQ(read, "7");
    EXPECT_EQ(8, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, ReadNBytesRandomAccessFile) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    tstring read;
    BufferedInputStream in(file.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(7, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "789");
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, SkipNBytesRandomAccessFile) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    tstring read;
    BufferedInputStream in(file.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(3));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(2, &read));
    EXPECT_EQ(read, "34");
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(2));
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(1, &read));
    EXPECT_EQ(read, "7");
    EXPECT_EQ(8, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, Seek) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    tstring read;
    BufferedInputStream in(input_stream.get(), buf_size);

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
}

TEST(BufferedInputStream, Seek_NotReset) {
  // This test verifies seek backwards within the buffer doesn't reset
  // input_stream
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file.get()));
  tstring read;
  BufferedInputStream in(input_stream.get(), 3);

  TF_ASSERT_OK(in.ReadNBytes(4, &read));
  int before_tell = input_stream->Tell();
  EXPECT_EQ(before_tell, 6);
  // Seek backwards
  TF_ASSERT_OK(in.Seek(3));
  int after_tell = input_stream->Tell();
  EXPECT_EQ(before_tell, after_tell);
}

TEST(BufferedInputStream, ReadAll_Empty) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  const std::string expected = "";
  TF_ASSERT_OK(WriteStringToFile(env, fname, expected));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    RandomAccessInputStream input_stream(file.get());
    BufferedInputStream in(&input_stream, buf_size);
    std::string contents;
    TF_ASSERT_OK(in.ReadAll(&contents));
    EXPECT_EQ(expected, contents);
  }
}

TEST(BufferedInputStream, ReadAll_Text) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  const std::string expected = "line one\nline two\nline three";
  TF_ASSERT_OK(WriteStringToFile(env, fname, expected));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    RandomAccessInputStream input_stream(file.get());
    BufferedInputStream in(&input_stream, buf_size);
    std::string contents;
    TF_ASSERT_OK(in.ReadAll(&contents));
    EXPECT_EQ(expected, contents);
  }
}

class PatternFile : public RandomAccessFile {
 public:
  explicit PatternFile(uint64_t size, bool short_ok = false,
                       bool external_view = false)
      : size_(size), short_ok_(short_ok), external_view_(external_view) {}

  absl::Status Read(uint64_t offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    ++read_calls_;
    if (offset >= size_) {
      *result = absl::string_view();
      return short_ok_ ? absl::OkStatus() : absl::OutOfRangeError("eof");
    }
    size_t count = static_cast<size_t>(std::min<uint64_t>(n, size_ - offset));
    if (short_ok_ && count > 3) {
      count = 3;
    }
    if (external_view_ && count <= backing_.size()) {
      *result = absl::string_view(backing_).substr(0, count);
    } else {
      for (size_t i = 0; i < count; ++i) {
        scratch[i] = static_cast<char>(((offset + i) * 131 + 17) & 0xFF);
      }
      *result = absl::string_view(scratch, count);
    }
    return (count < n && !short_ok_) ? absl::OutOfRangeError("eof")
                                     : absl::OkStatus();
  }

  int read_calls() const { return read_calls_; }

 private:
  uint64_t size_;
  bool short_ok_;
  bool external_view_;
  std::string backing_ = "external_bytes";
  mutable int read_calls_ = 0;
};

class OversizedSliceFile : public RandomAccessFile {
 public:
  absl::Status Read(uint64_t offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    *result = payload_;
    return absl::OkStatus();
  }

 private:
  std::string payload_ = std::string(64, 'z');
};

class NoTellInputStream : public InputStreamInterface {
 public:
  explicit NoTellInputStream(std::string data) : data_(std::move(data)) {}

  absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) override {
    if (bytes_to_read < 0) {
      return absl::InvalidArgumentError("negative");
    }
    const size_t n = static_cast<size_t>(
        std::min<uint64_t>(bytes_to_read, data_.size() - pos_));
    result->assign(data_.data() + pos_, n);
    pos_ += n;
    return n < static_cast<uint64_t>(bytes_to_read)
               ? absl::OutOfRangeError("eof")
               : absl::OkStatus();
  }

  int64_t Tell() const override { return -1; }

  absl::Status Reset() override {
    pos_ = 0;
    ++resets_;
    return absl::OkStatus();
  }

  int resets() const { return resets_; }

 private:
  std::string data_;
  size_t pos_ = 0;
  int resets_ = 0;
};

class ZeroByteOkInputStream : public InputStreamInterface {
 public:
  explicit ZeroByteOkInputStream(std::string data) : data_(std::move(data)) {}

  absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) override {
    if (pos_ >= data_.size()) {
      result->clear();
      return absl::OkStatus();
    }
    const size_t n = static_cast<size_t>(
        std::min<uint64_t>(bytes_to_read, data_.size() - pos_));
    result->assign(data_.data() + pos_, n);
    pos_ += n;
    return absl::OkStatus();
  }

  int64_t Tell() const override { return static_cast<int64_t>(pos_); }

  absl::Status Reset() override {
    pos_ = 0;
    return absl::OkStatus();
  }

 private:
  std::string data_;
  size_t pos_ = 0;
};

class DirtyErrorInputStream : public InputStreamInterface {
 public:
  absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) override {
    if (calls_++ == 0) {
      result->assign("abc", 3);
      return absl::OkStatus();
    }
    // Intentionally leave *result untouched on failure.
    return absl::InternalError("read failed");
  }

  int64_t Tell() const override { return 3; }

  absl::Status Reset() override {
    calls_ = 0;
    return absl::OkStatus();
  }

 private:
  int calls_ = 0;
};

TEST(BufferedInputStream, ReadAllDrainsUnreadBuffer) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  const std::string content = "line one\nline two\nline three";
  TF_ASSERT_OK(WriteStringToFile(env, fname, content));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (int buf_size : BufferSizes()) {
    {
      BufferedInputStream in(file.get(), buf_size);
      std::string line;
      TF_ASSERT_OK(in.ReadLine(&line));
      EXPECT_EQ(line, "line one");
      std::string rest = "stale";
      TF_ASSERT_OK(in.ReadAll(&rest));
      EXPECT_EQ(rest, "line two\nline three") << "buf_size=" << buf_size;
      EXPECT_EQ(in.Tell(), static_cast<int64_t>(content.size()));
      TF_ASSERT_OK(in.ReadAll(&rest));
      EXPECT_EQ(rest, "");
    }
    {
      BufferedInputStream in(file.get(), buf_size);
      tstring prefix;
      TF_ASSERT_OK(in.ReadNBytes(5, &prefix));
      EXPECT_EQ(prefix, "line ");
      tstring rest;
      TF_ASSERT_OK(in.ReadAll(&rest));
      EXPECT_EQ(rest, content.substr(5)) << "buf_size=" << buf_size;
    }
    {
      BufferedInputStream in(file.get(), buf_size);
      TF_ASSERT_OK(in.SkipLine());
      std::string rest;
      TF_ASSERT_OK(in.ReadAll(&rest));
      EXPECT_EQ(rest, "line two\nline three") << "buf_size=" << buf_size;
    }
    {
      BufferedInputStream in(file.get(), buf_size);
      tstring prefix;
      TF_ASSERT_OK(in.ReadNBytes(12, &prefix));
      TF_ASSERT_OK(in.Seek(4));
      std::string rest;
      TF_ASSERT_OK(in.ReadAll(&rest));
      EXPECT_EQ(rest, content.substr(4)) << "buf_size=" << buf_size;
    }
  }

  // Must not issue a redundant underlying ReadNBytes after cached OutOfRange.
  ReadOnceInputStream once;
  BufferedInputStream in(&once, 64);
  tstring prefix;
  TF_ASSERT_OK(in.ReadNBytes(3, &prefix));
  EXPECT_EQ(prefix, "012");
  std::string rest;
  TF_ASSERT_OK(in.ReadAll(&rest));
  EXPECT_EQ(rest, "3456789");
  EXPECT_EQ(in.Tell(), 10);
  TF_ASSERT_OK(in.ReadAll(&rest));
  EXPECT_EQ(rest, "");
  EXPECT_TRUE(absl::IsOutOfRange(in.SkipNBytes(2)));
}

TEST(BufferedInputStream, ZeroAndMaxBufferBytes) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  const std::string content = "alpha\nbeta\ngamma";
  TF_ASSERT_OK(WriteStringToFile(env, fname, content));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  {
    BufferedInputStream in(file.get(), 0);
    tstring chunk;
    TF_ASSERT_OK(in.ReadNBytes(6, &chunk));
    EXPECT_EQ(chunk, "alpha\n");
    std::string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "beta");
    tstring tline;
    TF_ASSERT_OK(in.ReadLine(&tline));
    EXPECT_EQ(tline, "gamma");
    EXPECT_EQ(in.ReadLineAsString(), "");
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
    TF_ASSERT_OK(in.Reset());
    EXPECT_EQ(in.ReadLineAsString(), "alpha\n");
    TF_ASSERT_OK(in.SkipLine());
    std::string rest;
    TF_ASSERT_OK(in.ReadAll(&rest));
    EXPECT_EQ(rest, "gamma");
  }

  for (size_t huge_buf :
       {static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        static_cast<size_t>(std::numeric_limits<int64_t>::max()) + size_t{1},
        std::numeric_limits<size_t>::max()}) {
    BufferedInputStream in(file.get(), huge_buf);
    tstring chunk;
    TF_ASSERT_OK(in.ReadNBytes(5, &chunk));
    EXPECT_EQ(chunk, "alpha");
    std::string rest;
    TF_ASSERT_OK(in.ReadAll(&rest));
    EXPECT_EQ(rest, "\nbeta\ngamma");
  }
}

TEST(BufferedInputStream, ReadNBytesInt64MaxAndMultiChunk) {
  PatternFile small_file(128);
  {
    BufferedInputStream in(&small_file, 64);
    tstring out;
    EXPECT_TRUE(absl::IsOutOfRange(
        in.ReadNBytes(std::numeric_limits<int64_t>::max(), &out)));
    ASSERT_EQ(out.size(), 128);
    EXPECT_LT(out.capacity(), 4096);
    for (size_t i = 0; i < out.size(); ++i) {
      ASSERT_EQ(out[i], static_cast<char>((i * 131 + 17) & 0xFF));
    }
  }

  constexpr size_t kLargeSize = 17 * 1024 * 1024 + 37;
  PatternFile large_file(kLargeSize);
  BufferedInputStream in(&large_file, 64 * 1024);
  tstring out;
  TF_ASSERT_OK(in.ReadNBytes(kLargeSize, &out));
  ASSERT_EQ(out.size(), kLargeSize);
  for (size_t i :
       {size_t{0}, size_t{1}, size_t{8 * 1024 * 1024}, kLargeSize - 1}) {
    EXPECT_EQ(out[i], static_cast<char>((i * 131 + 17) & 0xFF));
  }
}

TEST(BufferedInputStream, SkipNBytesTellSeekOverflowAndBoundaries) {
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  // Exact buffer-boundary skip and seek stay in-buffer without resetting or
  // advancing the underlying stream.
  {
    RandomAccessInputStream raw(file.get());
    BufferedInputStream in(&raw, 4);
    tstring out;
    TF_ASSERT_OK(in.ReadNBytes(1, &out));
    EXPECT_EQ(out, "0");
    EXPECT_EQ(raw.Tell(), 4);
    TF_ASSERT_OK(in.SkipNBytes(3));
    EXPECT_EQ(in.Tell(), 4);
    EXPECT_EQ(raw.Tell(), 4);
    TF_ASSERT_OK(in.Seek(4));
    TF_ASSERT_OK(in.Seek(0));
    EXPECT_EQ(raw.Tell(), 4);
    TF_ASSERT_OK(in.Seek(4));
    EXPECT_EQ(raw.Tell(), 4);
    TF_ASSERT_OK(in.Seek(1));
    TF_ASSERT_OK(in.ReadNBytes(3, &out));
    EXPECT_EQ(out, "123");
    EXPECT_TRUE(
        absl::IsOutOfRange(in.SkipNBytes(std::numeric_limits<int64_t>::max())));
    EXPECT_EQ(in.Tell(), 10);
  }

  // Streams returning Tell() == -1 preserve -1 and support Seek without
  // signed overflow or offset drift.
  {
    NoTellInputStream raw("0123456789");
    BufferedInputStream in(&raw, 4);
    EXPECT_EQ(in.Tell(), -1);
    tstring out;
    TF_ASSERT_OK(in.ReadNBytes(2, &out));
    EXPECT_EQ(out, "01");
    EXPECT_EQ(in.Tell(), -1);
    TF_ASSERT_OK(in.Seek(5));
    EXPECT_EQ(raw.resets(), 1);
    TF_ASSERT_OK(in.ReadNBytes(3, &out));
    EXPECT_EQ(out, "567");
    EXPECT_TRUE(
        absl::IsOutOfRange(in.Seek(std::numeric_limits<int64_t>::max())));
    EXPECT_EQ(in.Tell(), -1);
  }
}

TEST(BufferedInputStream, ZeroByteOkDirtyErrorAndNullGuards) {
  {
    ZeroByteOkInputStream raw("hi\nthere");
    BufferedInputStream in(&raw, 4);
    std::string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "hi");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "there");
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadLine(&line)));
    EXPECT_TRUE(absl::IsOutOfRange(in.SkipLine()));
    TF_ASSERT_OK(in.Reset());
    tstring out;
    EXPECT_TRUE(absl::IsOutOfRange(in.ReadNBytes(16, &out)));
    EXPECT_EQ(out, "hi\nthere");
  }

  {
    DirtyErrorInputStream raw;
    BufferedInputStream in(&raw, 3);
    tstring out;
    EXPECT_TRUE(absl::IsInternal(in.ReadNBytes(6, &out)));
    EXPECT_EQ(out, "abc");
    EXPECT_TRUE(absl::IsInternal(in.ReadNBytes(1, &out)));
    EXPECT_EQ(out, "");
  }

  {
    BufferedInputStream null_in(static_cast<InputStreamInterface*>(nullptr), 8);
    tstring out = "stale";
    std::string s = "stale";
    EXPECT_EQ(null_in.Tell(), -1);
    EXPECT_TRUE(absl::IsInvalidArgument(null_in.ReadNBytes(4, &out)));
    EXPECT_EQ(out, "");
    EXPECT_TRUE(absl::IsInvalidArgument(null_in.ReadNBytes(0, &out)));
    EXPECT_TRUE(absl::IsInvalidArgument(null_in.ReadNBytes(4, nullptr)));
    EXPECT_TRUE(absl::IsInvalidArgument(null_in.ReadLine(&s)));
    EXPECT_TRUE(absl::IsInvalidArgument(
        null_in.ReadLine(static_cast<std::string*>(nullptr))));
    EXPECT_TRUE(absl::IsInvalidArgument(
        null_in.ReadAll(static_cast<std::string*>(nullptr))));
    EXPECT_TRUE(absl::IsInvalidArgument(null_in.ReadAll(&s)));
    EXPECT_EQ(null_in.ReadLineAsString(), "");
    EXPECT_TRUE(absl::IsInvalidArgument(null_in.SkipLine()));
    EXPECT_TRUE(absl::IsInvalidArgument(null_in.SkipNBytes(1)));
    EXPECT_TRUE(absl::IsInvalidArgument(null_in.Seek(0)));
    EXPECT_TRUE(absl::IsInvalidArgument(null_in.Reset()));

    BufferedInputStream null_file(static_cast<RandomAccessFile*>(nullptr), 8);
    EXPECT_EQ(null_file.Tell(), -1);
    EXPECT_TRUE(absl::IsInvalidArgument(null_file.ReadNBytes(1, &out)));
  }
}

TEST(BufferedInputStream, RandomAccessInputStreamInvalidSlicesAndInt64Max) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  tstring out;

  // External non-scratch slice and short-OK single vs multi-chunk handling.
  PatternFile ext_file(6, /*short_ok=*/false, /*external_view=*/true);
  RandomAccessInputStream ext_in(&ext_file);
  TF_ASSERT_OK(ext_in.ReadNBytes(6, &out));
  EXPECT_EQ(out, "extern");
  EXPECT_EQ(ext_in.Tell(), 6);

  PatternFile short_ok(10, /*short_ok=*/true);
  RandomAccessInputStream short_in(&short_ok);
  TF_ASSERT_OK(short_in.ReadNBytes(5, &out));
  EXPECT_EQ(out.size(), 3);
  EXPECT_TRUE(absl::IsOutOfRange(short_in.ReadNBytes(kMax, &out)));
  EXPECT_EQ(out.size(), 3);
  EXPECT_EQ(short_in.Tell(), 6);

  // Oversized Read slice rejected without heap buffer overflow.
  OversizedSliceFile oversized;
  RandomAccessInputStream bad_in(&oversized);
  EXPECT_TRUE(absl::IsInternal(bad_in.ReadNBytes(4, &out)));
  EXPECT_EQ(out, "");
  EXPECT_EQ(bad_in.Tell(), 0);
  EXPECT_TRUE(absl::IsInternal(bad_in.SkipNBytes(4)));
  EXPECT_EQ(bad_in.Tell(), 0);
  TF_ASSERT_OK(bad_in.Seek(kMax - 2));
  EXPECT_TRUE(absl::IsInternal(bad_in.SkipNBytes(5)));
  EXPECT_EQ(bad_in.Tell(), kMax - 2);

  // In-bounds INT64_MAX skip uses a single 1-byte probe without overflow.
  PatternFile max_file(static_cast<uint64_t>(kMax));
  RandomAccessInputStream max_in(&max_file);
  TF_ASSERT_OK(max_in.SkipNBytes(kMax));
  EXPECT_EQ(max_in.Tell(), kMax);
  EXPECT_EQ(max_file.read_calls(), 1);
  TF_ASSERT_OK(max_in.SkipNBytes(0));
  EXPECT_TRUE(absl::IsOutOfRange(max_in.SkipNBytes(1)));
  EXPECT_TRUE(absl::IsOutOfRange(max_in.SkipNBytes(kMax)));
  TF_ASSERT_OK(max_in.Seek(kMax - 5));
  TF_ASSERT_OK(max_in.SkipNBytes(5));
  EXPECT_EQ(max_in.Tell(), kMax);
  TF_ASSERT_OK(max_in.Seek(kMax - 5));
  EXPECT_TRUE(absl::IsOutOfRange(max_in.SkipNBytes(6)));
  EXPECT_EQ(max_in.Tell(), kMax);
  TF_ASSERT_OK(max_in.Seek(kMax - 5));
  EXPECT_TRUE(absl::IsOutOfRange(max_in.SkipNBytes(10)));
  EXPECT_EQ(max_in.Tell(), kMax);

  // Virtual file exceeding INT64_MAX clamps offset and output cleanly.
  PatternFile huge_file(static_cast<uint64_t>(kMax) + 16);
  RandomAccessInputStream huge_in(&huge_file);
  TF_ASSERT_OK(huge_in.Seek(kMax - 2));
  TF_ASSERT_OK(huge_in.ReadNBytes(2, &out));
  EXPECT_EQ(out.size(), 2);
  EXPECT_EQ(huge_in.Tell(), kMax);
  TF_ASSERT_OK(huge_in.Seek(kMax - 2));
  EXPECT_TRUE(absl::IsOutOfRange(huge_in.ReadNBytes(5, &out)));
  EXPECT_EQ(out.size(), 2);
  EXPECT_EQ(huge_in.Tell(), kMax);
  TF_ASSERT_OK(huge_in.Seek(kMax - 2));
  EXPECT_TRUE(absl::IsOutOfRange(huge_in.SkipNBytes(5)));
  EXPECT_EQ(huge_in.Tell(), kMax);
}

void BM_BufferedReaderSmallReads(::testing::benchmark::State& state) {
  const int buff_size = state.range(0);
  const int file_size = state.range(1);
  Env* env = Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));

  const std::string file_elem = "0123456789";
  std::unique_ptr<WritableFile> write_file;
  TF_ASSERT_OK(env->NewWritableFile(fname, &write_file));
  for (int i = 0; i < file_size; ++i) {
    TF_ASSERT_OK(write_file->Append(file_elem));
  }
  TF_ASSERT_OK(write_file->Close());

  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  tstring result;

  int itr = 0;
  for (auto s : state) {
    BufferedInputStream in(file.get(), buff_size);
    for (int64_t i = 0; i < 10 * file_size; ++i) {
      TF_ASSERT_OK(in.ReadNBytes(1, &result))
          << "i: " << i << " itr: " << itr << " buff_size: " << buff_size
          << " file size: " << file_size;
    }
    ++itr;
  }
}
BENCHMARK(BM_BufferedReaderSmallReads)
    ->ArgPair(1, 5)
    ->ArgPair(1, 1024)
    ->ArgPair(10, 5)
    ->ArgPair(10, 1024)
    ->ArgPair(1024, 1024)
    ->ArgPair(1024 * 1024, 1024)
    ->ArgPair(1024 * 1024, 1024 * 1024)
    ->ArgPair(256 * 1024 * 1024, 1024);

}  // anonymous namespace
}  // namespace io
}  // namespace tsl
