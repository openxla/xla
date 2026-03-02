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

#ifndef XLA_TSL_PLATFORM_CLOUD_S3_FILE_SYSTEM_H_
#define XLA_TSL_PLATFORM_CLOUD_S3_FILE_SYSTEM_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/cloud/http_request.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/retrying_file_system.h"

namespace tsl {

/// AWS S3 implementation of a file system.
///
/// This provides basic S3 support for XLA operations that need to read/write
/// files on S3 (e.g., the per-fusion autotune cache). It uses the same
/// HttpRequest infrastructure as the GCS filesystem and implements AWS
/// Signature Version 4 for authentication.
///
/// Authentication is handled via standard AWS credential sources:
///   - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
///   - AWS credential files (~/.aws/credentials)
///   - EC2 instance metadata (IMDS)
///   - ECS task role credentials
///
/// The S3 region is determined from:
///   - AWS_REGION or AWS_DEFAULT_REGION environment variables
///   - Defaults to "us-east-1"
///
/// Custom S3 endpoints can be set via:
///   - S3_ENDPOINT environment variable (useful for MinIO, LocalStack, etc.)
///
/// The clients should use RetryingS3FileSystem defined below,
/// which adds retry logic to S3 operations.
class S3FileSystem : public FileSystem {
 public:
  S3FileSystem();
  S3FileSystem(std::unique_ptr<HttpRequest::Factory> http_request_factory);

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  absl::Status NewRandomAccessFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override;

  absl::Status NewWritableFile(const std::string& fname,
                               TransactionToken* token,
                               std::unique_ptr<WritableFile>* result) override;

  absl::Status NewAppendableFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) override;

  absl::Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  absl::Status FileExists(const std::string& fname,
                          TransactionToken* token) override;

  absl::Status Stat(const std::string& fname, TransactionToken* token,
                    FileStatistics* stat) override;

  absl::Status GetChildren(const std::string& dir, TransactionToken* token,
                           std::vector<std::string>* result) override;

  absl::Status GetMatchingPaths(const std::string& pattern,
                                TransactionToken* token,
                                std::vector<std::string>* results) override;

  absl::Status DeleteFile(const std::string& fname,
                          TransactionToken* token) override;

  absl::Status CreateDir(const std::string& dirname,
                         TransactionToken* token) override;

  absl::Status DeleteDir(const std::string& dirname,
                         TransactionToken* token) override;

  absl::Status GetFileSize(const std::string& fname, TransactionToken* token,
                           uint64* file_size) override;

  absl::Status RenameFile(const std::string& src, const std::string& target,
                          TransactionToken* token) override;

  absl::Status IsDirectory(const std::string& fname,
                           TransactionToken* token) override;

  absl::Status DeleteRecursively(const std::string& dirname,
                                 TransactionToken* token,
                                 int64_t* undeleted_files,
                                 int64_t* undeleted_dirs) override;

  void FlushCaches(TransactionToken* token) override;

  absl::Status HasAtomicMove(const std::string& path,
                             bool* has_atomic_move) override;

  /// Splits an S3 path into bucket and object components.
  /// For example, "s3://my-bucket/path/to/file" splits into
  /// bucket="my-bucket" and object="path/to/file".
  absl::Status ParseS3Path(absl::string_view fname, bool empty_object_ok,
                            std::string* bucket, std::string* object);

  /// Computes HMAC-SHA256 (exposed for testing).
  static std::string HmacSha256(const std::string& key,
                                const std::string& data);

  /// Computes SHA256 hex digest (exposed for testing).
  static std::string Sha256Hex(const std::string& data);

 private:
  /// Creates an HttpRequest with AWS SigV4 authentication headers.
  absl::Status CreateSignedRequest(
      std::unique_ptr<HttpRequest>* request, const std::string& method,
      const std::string& bucket, const std::string& object,
      const std::string& content_sha256 = "UNSIGNED-PAYLOAD");

  /// Loads AWS credentials from the environment or credential files.
  absl::Status LoadCredentials();

  /// Gets the S3 endpoint URL for a given bucket.
  std::string GetEndpointUrl(const std::string& bucket,
                             const std::string& object);

  /// Computes AWS Signature V4.
  std::string ComputeSignatureV4(
      const std::string& method, const std::string& canonical_uri,
      const std::string& canonical_querystring,
      const std::string& signed_headers,
      const std::string& canonical_headers_str,
      const std::string& payload_hash, const std::string& datestamp,
      const std::string& amz_date);

  std::shared_ptr<HttpRequest::Factory> http_request_factory_;

  absl::Mutex mu_;
  std::string access_key_ ABSL_GUARDED_BY(mu_);
  std::string secret_key_ ABSL_GUARDED_BY(mu_);
  std::string session_token_ ABSL_GUARDED_BY(mu_);

  std::string region_;
  std::string endpoint_;  // Custom endpoint, e.g. for MinIO/LocalStack.

  S3FileSystem(const S3FileSystem&) = delete;
  void operator=(const S3FileSystem&) = delete;
};

/// S3 implementation of a file system with retry on failures.
class RetryingS3FileSystem : public RetryingFileSystem<S3FileSystem> {
 public:
  RetryingS3FileSystem();
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_CLOUD_S3_FILE_SYSTEM_H_
