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
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/io/random_inputstream.h"

namespace tsl {
namespace io {
namespace {
constexpr size_t kMaxInitialReserve = 8 * 1024 * 1024;
constexpr size_t kMaxTStringSize = std::numeric_limits<size_t>::max() >> 2;
constexpr uint64_t kMaxBufSize = std::min<uint64_t>(
    std::numeric_limits<int64_t>::max(), std::numeric_limits<size_t>::max());
}  // namespace

BufferedInputStream::BufferedInputStream(InputStreamInterface* input_stream,
                                         size_t buffer_bytes,
                                         bool owns_input_stream)
    : input_stream_(input_stream),
      size_(std::clamp<uint64_t>(buffer_bytes, 1, kMaxBufSize)),
      owns_input_stream_(owns_input_stream) {
  buf_.reserve(std::min(size_, kMaxInitialReserve));
}

BufferedInputStream::BufferedInputStream(RandomAccessFile* file,
                                         size_t buffer_bytes)
    : BufferedInputStream(new RandomAccessInputStream(file), buffer_bytes,
                          true) {}

BufferedInputStream::~BufferedInputStream() {
  if (owns_input_stream_) {
    delete input_stream_;
  }
}

absl::Status BufferedInputStream::FillBuffer() {
  if (!input_stream_) {
    return absl::InvalidArgumentError("Null input stream");
  }
  if (!file_status_.ok()) {
    pos_ = 0;
    limit_ = 0;
    return file_status_;
  }
  buf_.clear();
  absl::Status s =
      input_stream_->ReadNBytes(static_cast<int64_t>(size_), &buf_);
  pos_ = 0;
  limit_ = buf_.size();
  if (s.ok() && limit_ == 0) {
    s = absl::OutOfRangeError("EOF reached");
  }
  if (!s.ok()) {
    file_status_ = s;
  }
  return s;
}

template <typename StringType>
absl::Status BufferedInputStream::ReadLineHelper(StringType* result,
                                                 bool include_eol) {
  if (result == nullptr) {
    return absl::InvalidArgumentError("Result is null");
  }
  result->clear();
  absl::Status s;
  size_t start_pos = pos_;
  auto append_span = [&]() {
    if (pos_ > start_pos) {
      result->append(std::as_const(buf_).data() + start_pos, pos_ - start_pos);
    }
  };
  while (true) {
    if (pos_ == limit_) {
      append_span();
      // Get more data into buffer
      s = FillBuffer();
      if (limit_ == 0) {
        break;
      }
      start_pos = pos_;
    }
    char c = std::as_const(buf_)[pos_];
    if (c == '\n') {
      append_span();
      if (include_eol) {
        result->append(1, c);
      }
      pos_++;
      return absl::OkStatus();
    }
    // We don't append '\r' to *result
    if (c == '\r') {
      append_span();
      start_pos = pos_ + 1;
    }
    pos_++;
  }
  if (absl::IsOutOfRange(s) && !result->empty()) {
    return absl::OkStatus();
  }
  return s;
}

absl::Status BufferedInputStream::ReadNBytes(int64_t bytes_to_read,
                                             tstring* result) {
  if (bytes_to_read < 0 || result == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid read request: ", bytes_to_read));
  }
  result->clear();
  if (!input_stream_ || bytes_to_read == 0 ||
      (pos_ == limit_ && !file_status_.ok())) {
    return !input_stream_ ? absl::InvalidArgumentError("Null input stream")
           : bytes_to_read == 0 ? absl::OkStatus()
                                : file_status_;
  }
  const uint64_t target = static_cast<uint64_t>(bytes_to_read);
  result->reserve(std::min<uint64_t>(target, kMaxInitialReserve));

  absl::Status s;
  while (result->size() < target) {
    // Check whether the buffer is fully read or not.
    if (pos_ == limit_) {
      s = FillBuffer();
      // If we didn't read any bytes, we're at the end of the file; break out.
      if (limit_ == 0) {
        DCHECK(!s.ok());
        file_status_ = s;
        break;
      }
    }
    const size_t bytes_to_copy =
        std::min<uint64_t>(limit_ - pos_, target - result->size());
    if (bytes_to_copy > kMaxTStringSize - result->size()) {
      s = absl::ResourceExhaustedError("Read exceeds maximum size");
      break;
    }
    const size_t cap = std::min<uint64_t>(target, kMaxTStringSize);
    if (const size_t need = result->size() + bytes_to_copy;
        need > result->capacity()) {
      result->reserve(std::max(
          need, result->capacity() <= cap / 2 ? result->capacity() * 2 : cap));
    }
    result->insert(result->size(), buf_, pos_, bytes_to_copy);
    pos_ += bytes_to_copy;
  }
  if (target > kMaxInitialReserve && result->size() < result->capacity() / 4) {
    tstring(std::as_const(*result).data(), result->size()).swap(*result);
  }
  // Filling the buffer might lead to a situation when we go past the end of
  // the file leading to an OutOfRange() status return. But we might have
  // obtained enough data to satisfy the function call. Returning OK then.
  if (absl::IsOutOfRange(s) && result->size() == target) {
    return absl::OkStatus();
  }
  return s;
}

absl::Status BufferedInputStream::SkipNBytes(int64_t bytes_to_skip) {
  if (bytes_to_skip < 0 || !input_stream_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Can only skip forward, not ", bytes_to_skip));
  }
  const size_t buffered = limit_ - pos_;
  if (static_cast<uint64_t>(bytes_to_skip) <= buffered) {
    pos_ += static_cast<size_t>(bytes_to_skip);
    return absl::OkStatus();
  }
  const int64_t remaining = bytes_to_skip - static_cast<int64_t>(buffered);
  pos_ = limit_ = 0;
  absl::Status s = absl::IsOutOfRange(file_status_)
                       ? file_status_
                       : input_stream_->SkipNBytes(remaining);
  if (absl::IsOutOfRange(s)) {
    file_status_ = s;
  }
  return s;
}

int64_t BufferedInputStream::Tell() const {
  const int64_t pos = input_stream_ ? input_stream_->Tell() : -1;
  const size_t buffered = limit_ - pos_;
  return (pos < 0 || buffered > static_cast<uint64_t>(pos))
             ? -1
             : pos - static_cast<int64_t>(buffered);
}

absl::Status BufferedInputStream::Seek(int64_t position) {
  if (position < 0 || !input_stream_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Seeking to an invalid position: ", position));
  }
  const int64_t pos = input_stream_->Tell();
  if (pos < 0 || limit_ > static_cast<uint64_t>(pos) ||
      position < pos - static_cast<int64_t>(limit_)) {
    ABSL_RETURN_IF_ERROR(Reset());
    return SkipNBytes(position);
  }
  const int64_t cur = pos - static_cast<int64_t>(limit_ - pos_);
  if (position <= cur) {
    pos_ -= static_cast<size_t>(cur - position);
    return absl::OkStatus();
  }
  return SkipNBytes(position - cur);
}

template <typename T>
absl::Status BufferedInputStream::ReadAll(T* result) {
  if (result == nullptr) {
    return absl::InvalidArgumentError("Result is null");
  }
  result->clear();
  if (pos_ < limit_) {
    result->append(std::as_const(buf_).data() + pos_, limit_ - pos_);
    pos_ = limit_;
  }
  if (!file_status_.ok()) {
    return absl::IsOutOfRange(file_status_) ? absl::OkStatus() : file_status_;
  }
  absl::Status status;
  while (status.ok()) {
    status = FillBuffer();
    if (limit_ == 0) {
      break;
    }
    result->append(std::as_const(buf_).data(), limit_);
    pos_ = limit_;
  }

  if (absl::IsOutOfRange(status)) {
    file_status_ = status;
    return absl::OkStatus();
  }
  return status;
}

template absl::Status BufferedInputStream::ReadAll<std::string>(
    std::string* result);
template absl::Status BufferedInputStream::ReadAll<tstring>(tstring* result);

absl::Status BufferedInputStream::Reset() {
  if (!input_stream_) {
    return absl::InvalidArgumentError("Null input stream");
  }
  ABSL_RETURN_IF_ERROR(input_stream_->Reset());
  pos_ = 0;
  limit_ = 0;
  file_status_ = absl::OkStatus();
  return absl::OkStatus();
}

absl::Status BufferedInputStream::ReadLine(std::string* result) {
  return ReadLineHelper(result, false);
}

absl::Status BufferedInputStream::ReadLine(tstring* result) {
  return ReadLineHelper(result, false);
}

std::string BufferedInputStream::ReadLineAsString() {
  std::string result;
  ReadLineHelper(&result, true).IgnoreError();
  return result;
}

absl::Status BufferedInputStream::SkipLine() {
  absl::Status s;
  bool skipped = false;
  while (true) {
    if (pos_ == limit_) {
      // Get more data into buffer
      s = FillBuffer();
      if (limit_ == 0) {
        break;
      }
    }
    char c = std::as_const(buf_)[pos_++];
    skipped = true;
    if (c == '\n') {
      return absl::OkStatus();
    }
  }
  if (absl::IsOutOfRange(s) && skipped) {
    return absl::OkStatus();
  }
  return s;
}

}  // namespace io
}  // namespace tsl
