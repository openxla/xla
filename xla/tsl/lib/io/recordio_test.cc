/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <limits>

#include "absl/status/status.h"
#include "tsl/platform/coding.h"
#include "tsl/platform/str_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/hash/crc32c.h"
#include "xla/tsl/lib/io/record_reader.h"
#include "xla/tsl/lib/io/record_writer.h"
#include "xla/tsl/lib/random/simple_philox.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace io {
namespace {

// Construct a string of the specified length made out of the supplied
// partial string.
std::string BigString(const std::string& partial_string, size_t n) {
  std::string result;
  while (result.size() < n) {
    result.append(partial_string);
  }
  result.resize(n);
  return result;
}

// Construct a string from a number
std::string NumberString(int n) {
  char buf[50];
  snprintf(buf, sizeof(buf), "%d.", n);
  return std::string(buf);
}

// Return a skewed potentially long string
std::string RandomSkewedString(int i, random::SimplePhilox* rnd) {
  return BigString(NumberString(i), rnd->Skewed(17));
}

class StringDest : public WritableFile {
 public:
  explicit StringDest(std::string* contents) : contents_(contents) {}

  absl::Status Close() override { return absl::OkStatus(); }
  absl::Status Flush() override { return absl::OkStatus(); }
  absl::Status Sync() override { return absl::OkStatus(); }
  absl::Status Append(absl::string_view slice) override {
    contents_->append(slice.data(), slice.size());
    return absl::OkStatus();
  }
#if defined(TF_CORD_SUPPORT)
  absl::Status Append(const absl::Cord& data) override {
    contents_->append(std::string(data));
    return absl::OkStatus();
  }
#endif
  absl::Status Tell(int64_t* pos) override {
    *pos = contents_->size();
    return absl::OkStatus();
  }

 private:
  std::string* contents_;
};

class StringSource : public RandomAccessFile {
 public:
  explicit StringSource(std::string* contents)
      : contents_(contents), force_error_(false) {}

  absl::Status Read(uint64_t offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    if (force_error_) {
      force_error_ = false;
      return absl::DataLossError("read error");
    }

    if (offset >= contents_->size()) {
      return absl::OutOfRangeError("end of file");
    }

    if (contents_->size() < offset + n) {
      n = contents_->size() - offset;
    }
    *result = absl::string_view(contents_->data() + offset, n);
    return absl::OkStatus();
  }

  void force_error() { force_error_ = true; }

 private:
  std::string* contents_;
  mutable bool force_error_;
};

class RecordioTest : public ::testing::Test {
 private:
  std::string contents_;
  StringDest dest_;
  StringSource source_;
  bool reading_;
  uint64_t readpos_;
  RecordWriter* writer_;
  RecordReader* reader_;

 public:
  RecordioTest()
      : dest_(&contents_),
        source_(&contents_),
        reading_(false),
        readpos_(0),
        writer_(new RecordWriter(&dest_)),
        reader_(new RecordReader(&source_)) {}

  ~RecordioTest() override {
    delete writer_;
    delete reader_;
  }

  void Write(const std::string& msg) {
    ASSERT_TRUE(!reading_) << "Write() after starting to read";
    TF_ASSERT_OK(writer_->WriteRecord(absl::string_view(msg)));
  }

#if defined(TF_CORD_SUPPORT)
  void Write(const absl::Cord& msg) {
    ASSERT_TRUE(!reading_) << "Write() after starting to read";
    TF_ASSERT_OK(writer_->WriteRecord(msg));
  }
#endif

  size_t WrittenBytes() const { return contents_.size(); }

  std::string Read() {
    if (!reading_) {
      reading_ = true;
    }
    tstring record;
    absl::Status s = reader_->ReadRecord(&readpos_, &record);
    if (s.ok()) {
      return record;
    } else if (absl::IsOutOfRange(s)) {
      return "EOF";
    } else {
      return s.ToString();
    }
  }

  void IncrementByte(int offset, int delta) { contents_[offset] += delta; }

  void SetByte(int offset, char new_byte) { contents_[offset] = new_byte; }

  void ShrinkSize(int bytes) { contents_.resize(contents_.size() - bytes); }

  void FixChecksum(int header_offset, int len) {
    // Compute crc of type/len/data
    uint32_t crc = crc32c::Value(&contents_[header_offset + 6], 1 + len);
    crc = crc32c::Mask(crc);
    core::EncodeFixed32(&contents_[header_offset], crc);
  }

  void ForceError() { source_.force_error(); }

  void StartReadingAt(uint64_t initial_offset) { readpos_ = initial_offset; }

  void CheckOffsetPastEndReturnsNoRecords(uint64_t offset_past_end) {
    Write("foo");
    Write("bar");
    Write(BigString("x", 10000));
    reading_ = true;
    uint64_t offset = WrittenBytes() + offset_past_end;
    tstring record;
    absl::Status s = reader_->ReadRecord(&offset, &record);
    ASSERT_TRUE(absl::IsOutOfRange(s)) << s;
  }
};

TEST_F(RecordioTest, Empty) { ASSERT_EQ("EOF", Read()); }

TEST(RecordReaderTest, RejectsWrappedRecordLength) {
  const uint64_t length = std::numeric_limits<uint64_t>::max() - 1;
  std::string contents(RecordReader::kHeaderSize + 2, '\0');
  core::EncodeFixed64(contents.data(), length);
  core::EncodeFixed32(
      contents.data() + sizeof(uint64_t),
      crc32c::Mask(crc32c::Value(contents.data(), sizeof(uint64_t))));

  StringSource metadata_source(&contents);
  RecordReader metadata_reader(&metadata_source);
  RecordReader::Metadata metadata;
  EXPECT_TRUE(absl::IsDataLoss(metadata_reader.GetMetadata(&metadata)));

  StringSource skip_source(&contents);
  RecordReader skip_reader(&skip_source);
  uint64_t offset = 0;
  int num_skipped = -1;
  EXPECT_TRUE(absl::IsDataLoss(
      skip_reader.SkipRecords(&offset, 1, &num_skipped)));
  EXPECT_EQ(offset, 0);
  EXPECT_EQ(num_skipped, 0);
}

// A reader positioned at kHeaderSize in a Snappy-compressed file decodes the
// payload of the first record as its own stream, so the payload can carry a
// record header that the writer would never emit.
namespace {
std::string MakeRecordHeader(uint64_t length) {
  std::string header(RecordReader::kHeaderSize, '\0');
  core::EncodeFixed64(header.data(), length);
  core::EncodeFixed32(
      header.data() + sizeof(uint64_t),
      crc32c::Mask(crc32c::Value(header.data(), sizeof(uint64_t))));
  return header;
}

std::string MakeSnappyFileWithPayload(const std::string& payload) {
  std::string contents;
  StringDest dst(&contents);
  RecordWriter writer(
      &dst, RecordWriterOptions::CreateRecordWriterOptions("SNAPPY"));
  EXPECT_TRUE(writer.WriteRecord(payload).ok());
  EXPECT_TRUE(writer.Close().ok());
  return contents;
}
}  // namespace

TEST(RecordReaderTest, SnappyReadRecordAcceptsGenuineRecord) {
  const RecordReaderOptions options =
      RecordReaderOptions::CreateRecordReaderOptions("SNAPPY");
  const std::string body = "abc";
  std::string footer(RecordReader::kFooterSize, '\0');
  core::EncodeFixed32(footer.data(),
                      crc32c::Mask(crc32c::Value(body.data(), body.size())));
  std::string contents = MakeSnappyFileWithPayload(
      MakeRecordHeader(body.size()) + body + footer);
  StringSource file(&contents);
  RecordReader reader(&file, options);
  uint64_t offset = RecordReader::kHeaderSize;
  tstring record;
  TF_ASSERT_OK(reader.ReadRecord(&offset, &record));
  EXPECT_EQ(record, body);
  EXPECT_EQ(offset, RecordReader::kHeaderSize + RecordReader::kHeaderSize +
                        body.size() + RecordReader::kFooterSize);
}

TEST(RecordReaderTest, SnappyReadRecordTruncatedLengthIsDataLoss) {
  const RecordReaderOptions options =
      RecordReaderOptions::CreateRecordReaderOptions("SNAPPY");
  std::string contents = MakeSnappyFileWithPayload(MakeRecordHeader(1000));
  StringSource file(&contents);
  RecordReader reader(&file, options);
  uint64_t offset = RecordReader::kHeaderSize;
  tstring record;
  EXPECT_TRUE(absl::IsDataLoss(reader.ReadRecord(&offset, &record)));
  EXPECT_EQ(offset, RecordReader::kHeaderSize);
}

TEST(RecordReaderTest, SnappyReadRecordRejectsLengthPastInt64) {
  const RecordReaderOptions options =
      RecordReaderOptions::CreateRecordReaderOptions("SNAPPY");
  const uint64_t int64_max = std::numeric_limits<int64_t>::max();
  // Every value is below SIZE_MAX - sizeof(uint32_t), so ReadChecksummed does
  // not reject it, but length + sizeof(uint32_t) exceeds INT64_MAX.
  for (uint64_t length : {int64_max, int64_max + 1,
                          std::numeric_limits<uint64_t>::max() - 5}) {
    std::string contents = MakeSnappyFileWithPayload(MakeRecordHeader(length));
    StringSource file(&contents);
    RecordReader reader(&file, options);
    uint64_t offset = RecordReader::kHeaderSize;
    tstring record;
    absl::Status s = reader.ReadRecord(&offset, &record);
    EXPECT_TRUE(absl::IsDataLoss(s)) << length << ": " << s;
    EXPECT_EQ(offset, RecordReader::kHeaderSize) << length;
  }
}

TEST_F(RecordioTest, ReadWrite) {
  Write("foo");
  Write("bar");
  Write("");
  Write("xxxx");
  ASSERT_EQ("foo", Read());
  ASSERT_EQ("bar", Read());
  ASSERT_EQ("", Read());
  ASSERT_EQ("xxxx", Read());
  ASSERT_EQ("EOF", Read());
  ASSERT_EQ("EOF", Read());  // Make sure reads at eof work
}

#if defined(TF_CORD_SUPPORT)
TEST_F(RecordioTest, ReadWriteCords) {
  Write(absl::Cord("foo"));
  Write(absl::Cord("bar"));
  Write(absl::Cord(""));
  Write(absl::Cord("xxxx"));
  ASSERT_EQ("foo", Read());
  ASSERT_EQ("bar", Read());
  ASSERT_EQ("", Read());
  ASSERT_EQ("xxxx", Read());
  ASSERT_EQ("EOF", Read());
  ASSERT_EQ("EOF", Read());  // Make sure reads at eof work
}
#endif

TEST_F(RecordioTest, ManyRecords) {
  for (int i = 0; i < 100000; i++) {
    Write(NumberString(i));
  }
  for (int i = 0; i < 100000; i++) {
    ASSERT_EQ(NumberString(i), Read());
  }
  ASSERT_EQ("EOF", Read());
}

TEST_F(RecordioTest, RandomRead) {
  const int N = 500;
  {
    random::PhiloxRandom philox(301, 17);
    random::SimplePhilox rnd(&philox);
    for (int i = 0; i < N; i++) {
      Write(RandomSkewedString(i, &rnd));
    }
  }
  {
    random::PhiloxRandom philox(301, 17);
    random::SimplePhilox rnd(&philox);
    for (int i = 0; i < N; i++) {
      ASSERT_EQ(RandomSkewedString(i, &rnd), Read());
    }
  }
  ASSERT_EQ("EOF", Read());
}

void TestNonSequentialReads(const RecordWriterOptions& writer_options,
                            const RecordReaderOptions& reader_options) {
  std::string contents;
  StringDest dst(&contents);
  RecordWriter writer(&dst, writer_options);
  for (int i = 0; i < 10; ++i) {
    TF_ASSERT_OK(writer.WriteRecord(NumberString(i))) << i;
  }
  TF_ASSERT_OK(writer.Close());

  StringSource file(&contents);
  RecordReader reader(&file, reader_options);

  tstring record;
  // First read sequentially to fill in the offsets table.
  uint64_t offsets[10] = {0};
  uint64_t offset = 0;
  for (int i = 0; i < 10; ++i) {
    offsets[i] = offset;
    TF_ASSERT_OK(reader.ReadRecord(&offset, &record)) << i;
  }

  // Read randomly: First go back to record #3 then forward to #8.
  offset = offsets[3];
  TF_ASSERT_OK(reader.ReadRecord(&offset, &record));
  EXPECT_EQ("3.", record);
  EXPECT_EQ(offsets[4], offset);

  offset = offsets[8];
  TF_ASSERT_OK(reader.ReadRecord(&offset, &record));
  EXPECT_EQ("8.", record);
  EXPECT_EQ(offsets[9], offset);
}

TEST_F(RecordioTest, NonSequentialReads) {
  TestNonSequentialReads(RecordWriterOptions(), RecordReaderOptions());
}

TEST_F(RecordioTest, NonSequentialReadsWithReadBuffer) {
  RecordReaderOptions options;
  options.buffer_size = 1 << 10;
  TestNonSequentialReads(RecordWriterOptions(), options);
}

TEST_F(RecordioTest, NonSequentialReadsWithCompression) {
  TestNonSequentialReads(
      RecordWriterOptions::CreateRecordWriterOptions("ZLIB"),
      RecordReaderOptions::CreateRecordReaderOptions("ZLIB"));
}

// Tests of all the error paths in log_reader.cc follow:
void AssertHasSubstr(absl::string_view s, absl::string_view expected) {
  EXPECT_TRUE(absl::StrContains(s, expected))
      << s << " does not contain " << expected;
}

void TestReadError(const RecordWriterOptions& writer_options,
                   const RecordReaderOptions& reader_options) {
  const std::string wrote = BigString("well hello there!", 100);
  std::string contents;
  StringDest dst(&contents);
  TF_ASSERT_OK(RecordWriter(&dst, writer_options).WriteRecord(wrote));

  StringSource file(&contents);
  RecordReader reader(&file, reader_options);

  uint64_t offset = 0;
  tstring read;
  file.force_error();
  absl::Status status = reader.ReadRecord(&offset, &read);
  ASSERT_TRUE(absl::IsDataLoss(status));
  ASSERT_EQ(0, offset);

  // A failed Read() shouldn't update the offset, and thus a retry shouldn't
  // lose the record.
  status = reader.ReadRecord(&offset, &read);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_GT(offset, 0);
  EXPECT_EQ(wrote, read);
}

TEST_F(RecordioTest, ReadError) {
  TestReadError(RecordWriterOptions(), RecordReaderOptions());
}

TEST_F(RecordioTest, ReadErrorWithBuffering) {
  RecordReaderOptions options;
  options.buffer_size = 1 << 20;
  TestReadError(RecordWriterOptions(), options);
}

TEST_F(RecordioTest, ReadErrorWithCompression) {
  TestReadError(RecordWriterOptions::CreateRecordWriterOptions("ZLIB"),
                RecordReaderOptions::CreateRecordReaderOptions("ZLIB"));
}

TEST_F(RecordioTest, CorruptLength) {
  Write("foo");
  IncrementByte(6, 100);
  AssertHasSubstr(Read(), "corrupted record");
}

TEST_F(RecordioTest, CorruptLengthCrc) {
  Write("foo");
  IncrementByte(10, 100);
  AssertHasSubstr(Read(), "corrupted record");
}

TEST_F(RecordioTest, CorruptData) {
  Write("foo");
  IncrementByte(14, 10);
  AssertHasSubstr(Read(), "corrupted record");
}

TEST_F(RecordioTest, CorruptDataCrc) {
  Write("foo");
  IncrementByte(WrittenBytes() - 1, 10);
  AssertHasSubstr(Read(), "corrupted record");
}

TEST_F(RecordioTest, ReadEnd) { CheckOffsetPastEndReturnsNoRecords(0); }

TEST_F(RecordioTest, ReadPastEnd) { CheckOffsetPastEndReturnsNoRecords(5); }

}  // namespace
}  // namespace io
}  // namespace tsl
