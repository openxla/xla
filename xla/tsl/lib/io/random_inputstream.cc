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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include "absl/status/status.h"

namespace tsl {
namespace io {
namespace {
constexpr int64_t kMaxSkipSize = 8 * 1024 * 1024;
constexpr int64_t kMaxReadChunk = 64 * 1024 * 1024;
constexpr int64_t kMaxPos = std::numeric_limits<int64_t>::max();
constexpr size_t kMaxTStringSize = std::numeric_limits<size_t>::max() >> 2;

bool AdvancePos(int64_t* pos, size_t n, size_t* applied = nullptr) {
  const uint64_t room = static_cast<uint64_t>(kMaxPos - *pos);
  *pos = n <= room ? *pos + static_cast<int64_t>(n) : kMaxPos;
  if (applied != nullptr && n > room) {
    *applied = static_cast<size_t>(room);
  }
  return n <= room;
}
}  // namespace

RandomAccessInputStream::RandomAccessInputStream(RandomAccessFile* file,
                                                 bool owns_file)
    : file_(file), owns_file_(owns_file) {}

RandomAccessInputStream::~RandomAccessInputStream() {
  if (owns_file_) {
    delete file_;
  }
}

absl::Status RandomAccessInputStream::ReadNBytes(int64_t bytes_to_read,
                                                 tstring* result) {
  if (bytes_to_read < 0 || !result) {
    return absl::InvalidArgumentError("Cannot read negative number of bytes");
  }
  result->clear();
  if (!file_ || pos_ < 0 || bytes_to_read == 0) {
    return (!file_ || pos_ < 0) ? absl::InvalidArgumentError("Invalid stream")
                                : absl::OkStatus();
  }

  const bool single_chunk = bytes_to_read <= kMaxSkipSize;
  const uint64_t target = static_cast<uint64_t>(bytes_to_read);
  const size_t cap = std::min<uint64_t>(target, kMaxTStringSize);
  size_t total = 0;
  int64_t step = kMaxSkipSize;
  absl::Status s;
  while (total < target) {
    if (total == kMaxTStringSize) {
      s = absl::ResourceExhaustedError("Read exceeds maximum size");
      break;
    }
    const size_t chunk = std::min<uint64_t>(
        target - total, std::min<uint64_t>(step, kMaxTStringSize - total));
    const size_t need = total + chunk;
    if (!single_chunk && need > result->capacity()) {
      result->reserve(std::max(
          need, result->capacity() <= cap / 2 ? result->capacity() * 2 : cap));
    }
    result->resize_uninitialized(need);
    char* dst = result->mdata() + total;
    absl::string_view data;
    s = file_->Read(pos_, data, {dst, chunk});
    if (data.size() > chunk || (!data.empty() && data.data() == nullptr)) {
      s = absl::InternalError("Invalid read slice");
      break;
    }
    if (!data.empty() && data.data() != dst) {
      std::memmove(dst, data.data(), data.size());
    }
    size_t applied = 0;
    if ((s.ok() || absl::IsOutOfRange(s)) &&
        !AdvancePos(&pos_, data.size(), &applied)) {
      total += applied;
      s = absl::OutOfRangeError("File offset overflow");
      break;
    }
    total += data.size();
    if (!s.ok() || data.size() < chunk) {
      s = (s.ok() && !single_chunk) ? absl::OutOfRangeError("EOF") : s;
      break;
    }
    step = std::min(step * 2, kMaxReadChunk);
  }
  result->resize(total);
  if (!single_chunk && total < result->capacity() / 4) {
    tstring(std::as_const(*result).data(), total).swap(*result);
  }
  return s;
}

#if defined(TF_CORD_SUPPORT)
absl::Status RandomAccessInputStream::ReadNBytes(int64_t bytes_to_read,
                                                 absl::Cord* result) {
  if (bytes_to_read < 0 || !result || !file_ || pos_ < 0 ||
      static_cast<uint64_t>(bytes_to_read) >
          std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError("Invalid read arguments");
  }
  if (bytes_to_read == 0) {
    return absl::OkStatus();
  }
  const size_t current_size = result->size();
  absl::Status s =
      file_->Read(pos_, static_cast<size_t>(bytes_to_read), result);
  if (s.ok() || absl::IsOutOfRange(s)) {
    if (result->size() < current_size ||
        result->size() - current_size > static_cast<size_t>(bytes_to_read)) {
      return absl::InternalError("Invalid Cord read size");
    }
    const size_t delta = result->size() - current_size;
    size_t applied = 0;
    if (!AdvancePos(&pos_, delta, &applied)) {
      result->RemoveSuffix(delta - applied);
      return absl::OutOfRangeError("File offset overflow");
    }
  }
  return s;
}
#endif

// To limit memory usage, the default implementation of SkipNBytes() only reads
// 8MB at a time.
absl::Status RandomAccessInputStream::SkipNBytes(int64_t bytes_to_skip) {
  if (bytes_to_skip < 0 || !file_ || pos_ < 0 || bytes_to_skip == 0) {
    return (bytes_to_skip == 0 && file_ && pos_ >= 0)
               ? absl::OkStatus()
               : absl::InvalidArgumentError("Invalid skip arguments");
  }
  if (bytes_to_skip <= kMaxPos - pos_) {
    char probe = 0;
    absl::string_view data;
    absl::Status s = file_->Read(pos_ + (bytes_to_skip - 1), data, {&probe, 1});
    if (data.size() > 1) {
      return absl::InternalError("Invalid read slice");
    }
    if ((s.ok() || absl::IsOutOfRange(s)) && data.size() == 1) {
      pos_ += bytes_to_skip;
      return absl::OkStatus();
    }
  }
  // Read kDefaultSkipSize at a time till bytes_to_skip.
  std::unique_ptr<char[]> scratch(
      new char[std::min<int64_t>(kMaxSkipSize, bytes_to_skip)]);
  while (bytes_to_skip > 0) {
    int64_t bytes_to_read = std::min<int64_t>(kMaxSkipSize, bytes_to_skip);
    absl::string_view data;
    absl::Status s = file_->Read(
        pos_, data, {scratch.get(), static_cast<size_t>(bytes_to_read)});
    if (data.size() > static_cast<size_t>(bytes_to_read) ||
        (!s.ok() && !absl::IsOutOfRange(s))) {
      return data.size() > static_cast<size_t>(bytes_to_read)
                 ? absl::InternalError("Invalid read slice")
                 : s;
    }
    if (!AdvancePos(&pos_, data.size())) {
      return absl::OutOfRangeError("File offset overflow");
    }
    if (data.size() < static_cast<size_t>(bytes_to_read)) {
      return absl::OutOfRangeError("reached end of file");
    }
    bytes_to_skip -= bytes_to_read;
  }
  return absl::OkStatus();
}

int64_t RandomAccessInputStream::Tell() const {
  return file_ == nullptr ? -1 : pos_;
}

}  // namespace io
}  // namespace tsl
