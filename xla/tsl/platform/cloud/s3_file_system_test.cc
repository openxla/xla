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

#include "xla/tsl/platform/cloud/s3_file_system.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/cloud/curl_http_request.h"
#include "xla/tsl/platform/cloud/http_request.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_statistics.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/retrying_utils.h"

#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#endif

namespace tsl {
namespace {

// A lenient fake HttpRequest that checks only the URI (not auth headers,
// which contain timestamps/signatures and change every call).
class LenientFakeHttpRequest : public CurlHttpRequest {
 public:
  // expected_uri_substr: a substring the URI must contain (for matching).
  // response: the response body to return.
  // response_headers: headers to return (e.g., Content-Length).
  // response_code: HTTP status code.
  LenientFakeHttpRequest(
      const std::string& expected_uri_substr, const std::string& response,
      const std::map<std::string, std::string>& response_headers = {},
      uint64_t response_code = 200,
      absl::Status response_status = absl::OkStatus())
      : expected_uri_substr_(expected_uri_substr),
        response_(response),
        response_headers_(response_headers),
        response_code_(response_code),
        response_status_(response_status) {}

  void SetUri(const std::string& uri) override { actual_uri_ = uri; }
  void SetRange(uint64_t start, uint64_t end) override {}
  void AddHeader(const std::string& name, const std::string& value) override {}
  void AddAuthBearerHeader(const std::string& auth_token) override {}
  void SetDeleteRequest() override { is_delete_ = true; }
  absl::Status SetPutFromFile(const std::string& body_filepath,
                              size_t offset) override {
    return absl::OkStatus();
  }
  void SetPostFromBuffer(const char* buffer, size_t size) override {
    post_body_ = std::string(buffer, size);
  }
  void SetPutEmptyBody() override { is_put_ = true; }
  void SetPostEmptyBody() override {}
  void SetResultBuffer(std::vector<char>* buffer) override {
    buffer->clear();
    buffer_ = buffer;
  }
  void SetResultBufferDirect(char* buffer, size_t size) override {
    direct_result_buffer_ = buffer;
    direct_result_buffer_size_ = size;
  }
  size_t GetResultBufferDirectBytesTransferred() override {
    return direct_result_bytes_transferred_;
  }

  absl::Status Send() override {
    if (!expected_uri_substr_.empty()) {
      EXPECT_TRUE(actual_uri_.find(expected_uri_substr_) != std::string::npos)
          << "Expected URI to contain '" << expected_uri_substr_
          << "', but got: " << actual_uri_;
    }
    if (buffer_) {
      buffer_->insert(buffer_->begin(), response_.data(),
                      response_.data() + response_.size());
    } else if (direct_result_buffer_ != nullptr) {
      size_t bytes_to_copy =
          std::min<size_t>(direct_result_buffer_size_, response_.size());
      memcpy(direct_result_buffer_, response_.data(), bytes_to_copy);
      direct_result_bytes_transferred_ += bytes_to_copy;
    }
    return response_status_;
  }

  std::string EscapeString(const std::string& str) override { return str; }

  std::string GetResponseHeader(const std::string& name) const override {
    auto it = response_headers_.find(name);
    return it != response_headers_.end() ? it->second : "";
  }

  uint64_t GetResponseCode() const override { return response_code_; }

  void SetTimeouts(uint32_t connection, uint32_t inactivity,
                   uint32_t total) override {}

  const std::string& actual_uri() const { return actual_uri_; }
  const std::string& post_body() const { return post_body_; }
  bool is_delete() const { return is_delete_; }
  bool is_put() const { return is_put_; }

 private:
  std::string expected_uri_substr_;
  std::string actual_uri_;
  std::string response_;
  std::map<std::string, std::string> response_headers_;
  uint64_t response_code_;
  absl::Status response_status_;
  std::vector<char>* buffer_ = nullptr;
  char* direct_result_buffer_ = nullptr;
  size_t direct_result_buffer_size_ = 0;
  size_t direct_result_bytes_transferred_ = 0;
  std::string post_body_;
  bool is_delete_ = false;
  bool is_put_ = false;
};

// A factory that returns pre-created LenientFakeHttpRequests in order.
class LenientFakeHttpRequestFactory : public HttpRequest::Factory {
 public:
  explicit LenientFakeHttpRequestFactory(
      std::vector<HttpRequest*>* requests)
      : requests_(requests) {}

  ~LenientFakeHttpRequestFactory() override {
    EXPECT_EQ(current_index_, requests_->size())
        << "Not all expected HTTP requests were consumed.";
  }

  HttpRequest* Create() override {
    EXPECT_LT(current_index_, requests_->size())
        << "Too many calls to HttpRequest factory.";
    return (*requests_)[current_index_++];
  }

 private:
  std::vector<HttpRequest*>* requests_;
  int current_index_ = 0;
};

// ===== ParseS3Path tests =====

TEST(S3FileSystemTest, ParseS3Path_BucketAndObject) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  std::string bucket, object;
  TF_EXPECT_OK(fs.ParseS3Path("s3://my-bucket/path/to/file.txt",
                               /*empty_object_ok=*/false, &bucket, &object));
  EXPECT_EQ(bucket, "my-bucket");
  EXPECT_EQ(object, "path/to/file.txt");
}

TEST(S3FileSystemTest, ParseS3Path_BucketOnly) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  std::string bucket, object;
  TF_EXPECT_OK(fs.ParseS3Path("s3://my-bucket", /*empty_object_ok=*/true,
                               &bucket, &object));
  EXPECT_EQ(bucket, "my-bucket");
  EXPECT_EQ(object, "");
}

TEST(S3FileSystemTest, ParseS3Path_BucketOnlyNotAllowed) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  std::string bucket, object;
  EXPECT_FALSE(fs.ParseS3Path("s3://my-bucket", /*empty_object_ok=*/false,
                               &bucket, &object)
                   .ok());
}

TEST(S3FileSystemTest, ParseS3Path_WrongScheme) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  std::string bucket, object;
  EXPECT_FALSE(
      fs.ParseS3Path("gs://bucket/obj", /*empty_object_ok=*/false,
                      &bucket, &object)
          .ok());
}

TEST(S3FileSystemTest, ParseS3Path_NoBucket) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  std::string bucket, object;
  EXPECT_FALSE(
      fs.ParseS3Path("s3://", /*empty_object_ok=*/true, &bucket, &object)
          .ok());
}

TEST(S3FileSystemTest, ParseS3Path_NestedPath) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  std::string bucket, object;
  TF_EXPECT_OK(fs.ParseS3Path("s3://bucket/a/b/c/d.txt",
                               /*empty_object_ok=*/false, &bucket, &object));
  EXPECT_EQ(bucket, "bucket");
  EXPECT_EQ(object, "a/b/c/d.txt");
}

TEST(S3FileSystemTest, ParseS3Path_TrailingSlash) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  std::string bucket, object;
  TF_EXPECT_OK(fs.ParseS3Path("s3://bucket/dir/",
                               /*empty_object_ok=*/false, &bucket, &object));
  EXPECT_EQ(bucket, "bucket");
  EXPECT_EQ(object, "dir/");
}

// ===== Sha256Hex tests =====

TEST(S3FileSystemTest, Sha256Hex_EmptyString) {
  // SHA256 of empty string is a well-known constant.
  std::string hash = S3FileSystem::Sha256Hex("");
  EXPECT_EQ(hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(S3FileSystemTest, Sha256Hex_KnownValue) {
  // SHA256 of "hello" is a well-known value.
  std::string hash = S3FileSystem::Sha256Hex("hello");
  EXPECT_EQ(hash,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
}

// ===== HmacSha256 tests =====

TEST(S3FileSystemTest, HmacSha256_NonEmpty) {
  // Just verify it returns a 32-byte result (SHA-256 output size).
  std::string result = S3FileSystem::HmacSha256("key", "data");
  EXPECT_EQ(result.size(), 32);
}

TEST(S3FileSystemTest, HmacSha256_Deterministic) {
  std::string r1 = S3FileSystem::HmacSha256("secret", "message");
  std::string r2 = S3FileSystem::HmacSha256("secret", "message");
  EXPECT_EQ(r1, r2);
}

TEST(S3FileSystemTest, HmacSha256_DifferentKeys) {
  std::string r1 = S3FileSystem::HmacSha256("key1", "data");
  std::string r2 = S3FileSystem::HmacSha256("key2", "data");
  EXPECT_NE(r1, r2);
}

// ===== CreateDir (no-op for S3) =====

TEST(S3FileSystemTest, CreateDir_IsNoOp) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  TF_EXPECT_OK(fs.CreateDir("s3://bucket/some/dir", nullptr));
}

TEST(S3FileSystemTest, CreateDir_BucketOnly) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  TF_EXPECT_OK(fs.CreateDir("s3://bucket", nullptr));
}

// ===== HasAtomicMove =====

TEST(S3FileSystemTest, HasAtomicMove_ReturnsFalse) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  bool has_atomic_move = true;
  TF_EXPECT_OK(fs.HasAtomicMove("s3://bucket/file", &has_atomic_move));
  EXPECT_FALSE(has_atomic_move);
}

// ===== FileExists with fake HTTP =====

TEST(S3FileSystemTest, FileExists_Found) {
  std::vector<HttpRequest*> requests({
      // HEAD request for the object — returns 200.
      new LenientFakeHttpRequest("my-bucket", "", {}, 200),
  });
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(
      new LenientFakeHttpRequestFactory(&requests)));
  TF_EXPECT_OK(fs.FileExists("s3://my-bucket/some/file.txt", nullptr));
}

TEST(S3FileSystemTest, FileExists_NotFound) {
  std::vector<HttpRequest*> requests({
      // HEAD request for the object — returns 404.
      new LenientFakeHttpRequest("my-bucket", "",
                                 {}, 404, absl::NotFoundError("not found")),
      // ListObjectsV2 fallback check — returns empty.
      new LenientFakeHttpRequest(
          "list-type=2", "<ListBucketResult><KeyCount>0</KeyCount>"
                         "</ListBucketResult>"),
  });
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(
      new LenientFakeHttpRequestFactory(&requests)));
  EXPECT_TRUE(absl::IsNotFound(
      fs.FileExists("s3://my-bucket/nonexistent.txt", nullptr)));
}

// ===== Stat with fake HTTP =====

TEST(S3FileSystemTest, Stat_File) {
  std::vector<HttpRequest*> requests({
      new LenientFakeHttpRequest(
          "my-bucket", "",
          {{"Content-Length", "12345"}}, 200),
  });
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(
      new LenientFakeHttpRequestFactory(&requests)));
  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("s3://my-bucket/file.txt", nullptr, &stat));
  EXPECT_EQ(stat.length, 12345);
  EXPECT_FALSE(stat.is_directory);
}

TEST(S3FileSystemTest, Stat_Bucket) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("s3://my-bucket", nullptr, &stat));
  EXPECT_TRUE(stat.is_directory);
  EXPECT_EQ(stat.length, 0);
}

// ===== GetChildren with fake HTTP =====

TEST(S3FileSystemTest, GetChildren_ListsObjects) {
  std::vector<HttpRequest*> requests({
      new LenientFakeHttpRequest(
          "list-type=2",
          "<ListBucketResult>"
          "<IsTruncated>false</IsTruncated>"
          "<Contents><Key>dir/file1.txt</Key></Contents>"
          "<Contents><Key>dir/file2.txt</Key></Contents>"
          "<CommonPrefixes><Prefix>dir/subdir/</Prefix></CommonPrefixes>"
          "</ListBucketResult>"),
  });
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(
      new LenientFakeHttpRequestFactory(&requests)));
  std::vector<std::string> children;
  TF_EXPECT_OK(fs.GetChildren("s3://my-bucket/dir", nullptr, &children));
  EXPECT_EQ(children.size(), 3);
  // Sort for deterministic comparison.
  std::sort(children.begin(), children.end());
  EXPECT_EQ(children[0], "file1.txt");
  EXPECT_EQ(children[1], "file2.txt");
  EXPECT_EQ(children[2], "subdir");
}

// ===== GetMatchingPaths with fake HTTP =====

TEST(S3FileSystemTest, GetMatchingPaths_GlobPattern) {
  std::vector<HttpRequest*> requests({
      new LenientFakeHttpRequest(
          "list-type=2",
          "<ListBucketResult>"
          "<IsTruncated>false</IsTruncated>"
          "<Contents><Key>cache/abc123.textproto</Key></Contents>"
          "<Contents><Key>cache/def456.textproto</Key></Contents>"
          "<Contents><Key>cache/readme.txt</Key></Contents>"
          "</ListBucketResult>"),
  });
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(
      new LenientFakeHttpRequestFactory(&requests)));
  std::vector<std::string> results;
  TF_EXPECT_OK(fs.GetMatchingPaths("s3://bucket/cache/*.textproto", nullptr,
                                   &results));
  EXPECT_EQ(results.size(), 2);
  std::sort(results.begin(), results.end());
  EXPECT_EQ(results[0], "s3://bucket/cache/abc123.textproto");
  EXPECT_EQ(results[1], "s3://bucket/cache/def456.textproto");
}

// ===== DeleteFile with fake HTTP =====

TEST(S3FileSystemTest, DeleteFile_Success) {
  std::vector<HttpRequest*> requests({
      new LenientFakeHttpRequest("my-bucket", "", {}, 204),
  });
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(
      new LenientFakeHttpRequestFactory(&requests)));
  TF_EXPECT_OK(fs.DeleteFile("s3://my-bucket/to-delete.txt", nullptr));
}

// ===== IsDirectory with fake HTTP =====

TEST(S3FileSystemTest, IsDirectory_BucketIsAlwaysDir) {
  // Bucket-level check doesn't need HTTP requests.
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  TF_EXPECT_OK(fs.IsDirectory("s3://my-bucket", nullptr));
}

TEST(S3FileSystemTest, IsDirectory_PrefixWithChildren) {
  std::vector<HttpRequest*> requests({
      new LenientFakeHttpRequest(
          "list-type=2",
          "<ListBucketResult><KeyCount>3</KeyCount>"
          "<Contents><Key>dir/file1.txt</Key></Contents>"
          "</ListBucketResult>"),
  });
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(
      new LenientFakeHttpRequestFactory(&requests)));
  TF_EXPECT_OK(fs.IsDirectory("s3://bucket/dir", nullptr));
}

TEST(S3FileSystemTest, IsDirectory_NotADirectory) {
  std::vector<HttpRequest*> requests({
      new LenientFakeHttpRequest(
          "list-type=2",
          "<ListBucketResult><KeyCount>0</KeyCount></ListBucketResult>"),
  });
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(
      new LenientFakeHttpRequestFactory(&requests)));
  EXPECT_TRUE(absl::IsFailedPrecondition(
      fs.IsDirectory("s3://bucket/not-a-dir", nullptr)));
}

// ===== Registration test =====

TEST(S3FileSystemTest, SchemeIsRegistered) {
  // Verify that the "s3" scheme is registered with the default Env.
  FileSystem* fs = nullptr;
  absl::Status status =
      Env::Default()->GetFileSystemForFile("s3://bucket/file", &fs);
  TF_EXPECT_OK(status);
  EXPECT_NE(fs, nullptr);
}

// ===== NewWritableFile creates a writable file =====

TEST(S3FileSystemTest, NewWritableFile_Creates) {
  S3FileSystem fs(std::unique_ptr<HttpRequest::Factory>(nullptr));
  std::unique_ptr<WritableFile> file;
  TF_EXPECT_OK(
      fs.NewWritableFile("s3://bucket/output.txt", nullptr, &file));
  EXPECT_NE(file, nullptr);
  absl::string_view name;
  TF_EXPECT_OK(file->Name(&name));
  EXPECT_EQ(name, "s3://bucket/output.txt");
}

}  // namespace
}  // namespace tsl
