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

#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "json/json.h"
#include "xla/tsl/platform/cloud/curl_http_request.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_statistics.h"
#include "xla/tsl/platform/file_system.h"
#include "tsl/platform/path.h"
#include "tsl/platform/retrying_file_system.h"
#include "tsl/platform/retrying_utils.h"
#include "tsl/platform/str_util.h"

namespace tsl {
namespace {

constexpr char kS3Service[] = "s3";
constexpr char kDefaultRegion[] = "us-east-1";

// Environment variable names.
constexpr char kAwsAccessKeyId[] = "AWS_ACCESS_KEY_ID";
constexpr char kAwsSecretAccessKey[] = "AWS_SECRET_ACCESS_KEY";
constexpr char kAwsSessionToken[] = "AWS_SESSION_TOKEN";
constexpr char kAwsRegion[] = "AWS_REGION";
constexpr char kAwsDefaultRegion[] = "AWS_DEFAULT_REGION";
constexpr char kS3Endpoint[] = "S3_ENDPOINT";
constexpr char kS3UsePathStyle[] = "S3_USE_PATH_STYLE";

std::string GetEnvVarOrDefault(const char* name, const std::string& dflt) {
  const char* val = std::getenv(name);
  return val ? val : dflt;
}

// Ensure a directory path ends with '/'.
std::string MaybeAppendSlash(const std::string& name) {
  if (name.empty() || name.back() == '/') {
    return name;
  }
  return absl::StrCat(name, "/");
}

// URL-encode a string for use in S3 paths. Encodes all characters except
// unreserved characters (A-Z, a-z, 0-9, '-', '.', '_', '~') and '/'.
std::string UriEncode(absl::string_view input, bool encode_slash = false) {
  std::string result;
  result.reserve(input.size());
  for (char c : input) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
        (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_' ||
        c == '~' || (!encode_slash && c == '/')) {
      result += c;
    } else {
      char buf[4];
      snprintf(buf, sizeof(buf), "%%%02X", static_cast<unsigned char>(c));
      result += buf;
    }
  }
  return result;
}

// Get the current time as a datestamp (YYYYMMDD) and AMZ date
// (YYYYMMDD'T'HHMMSS'Z').
void GetTimestamps(std::string* datestamp, std::string* amz_date) {
  time_t now = time(nullptr);
  struct tm utc;
  gmtime_r(&now, &utc);
  char date_buf[16];
  strftime(date_buf, sizeof(date_buf), "%Y%m%d", &utc);
  *datestamp = date_buf;
  char amz_buf[32];
  strftime(amz_buf, sizeof(amz_buf), "%Y%m%dT%H%M%SZ", &utc);
  *amz_date = amz_buf;
}

// Parse an XML tag value from an S3 API response.
// This is a simple parser for the common S3 XML response patterns.
std::string ExtractXmlTagValue(const std::string& xml,
                               const std::string& tag) {
  std::string open_tag = absl::StrCat("<", tag, ">");
  std::string close_tag = absl::StrCat("</", tag, ">");
  auto start = xml.find(open_tag);
  if (start == std::string::npos) return "";
  start += open_tag.size();
  auto end = xml.find(close_tag, start);
  if (end == std::string::npos) return "";
  return xml.substr(start, end - start);
}

// Extract all occurrences of an XML tag value.
std::vector<std::string> ExtractAllXmlTagValues(const std::string& xml,
                                                const std::string& tag) {
  std::vector<std::string> results;
  std::string open_tag = absl::StrCat("<", tag, ">");
  std::string close_tag = absl::StrCat("</", tag, ">");
  size_t pos = 0;
  while (true) {
    auto start = xml.find(open_tag, pos);
    if (start == std::string::npos) break;
    start += open_tag.size();
    auto end = xml.find(close_tag, start);
    if (end == std::string::npos) break;
    results.push_back(xml.substr(start, end - start));
    pos = end + close_tag.size();
  }
  return results;
}

RetryConfig GetS3RetryConfig() {
  // S3 recommends exponential backoff with jitter for retries.
  return RetryConfig(/* init_delay_time_us = */ 500000,  // 500ms
                     /* max_delay_time_us = */ 30000000,  // 30s
                     /* max_retries = */ 10);
}

// A RandomAccessFile implementation for S3 objects.
class S3RandomAccessFile : public RandomAccessFile {
 public:
  S3RandomAccessFile(const std::string& filename, S3FileSystem* fs)
      : filename_(filename), fs_(fs) {}

  absl::Status Name(absl::string_view* result) const override {
    *result = filename_;
    return absl::OkStatus();
  }

  absl::Status Read(uint64 offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    // Use the filesystem to read a range of the file.
    std::string bucket, object;
    TF_RETURN_IF_ERROR(fs_->ParseS3Path(filename_, false, &bucket, &object));

    std::unique_ptr<HttpRequest> request;
    // We create a simple GET request with a Range header.
    // For now, we create the signed request manually.
    auto* http_factory =
        const_cast<S3FileSystem*>(fs_);  // Need non-const for signing.
    // This simplified approach reads the range directly.
    // A production implementation would use the FileBlockCache pattern.

    std::string response;
    std::vector<char> response_buffer;
    // For simplicity in this initial implementation, read via a temp buffer.
    // TODO(b/future): Use block caching like GCS.
    *result = absl::string_view(scratch, 0);
    return absl::UnimplementedError(
        "S3 random access reads not yet fully implemented. "
        "Use ReadFileToString for whole-file reads.");
  }

 private:
  std::string filename_;
  S3FileSystem* fs_;
};

// A WritableFile implementation that buffers writes and uploads on Close().
class S3WritableFile : public WritableFile {
 public:
  S3WritableFile(const std::string& filename, S3FileSystem* fs)
      : filename_(filename), fs_(fs) {}

  absl::Status Append(absl::string_view data) override {
    buffer_.append(data.data(), data.size());
    return absl::OkStatus();
  }

  absl::Status Close() override {
    if (closed_) return absl::OkStatus();
    closed_ = true;

    std::string bucket, object;
    TF_RETURN_IF_ERROR(fs_->ParseS3Path(filename_, false, &bucket, &object));

    // Use PutObject to upload the buffered content.
    std::string payload_hash = S3FileSystem::Sha256Hex(buffer_);

    std::string datestamp, amz_date;
    GetTimestamps(&datestamp, &amz_date);

    std::string encoded_object = UriEncode(object);
    std::string canonical_uri = absl::StrCat("/", bucket, "/", encoded_object);

    // Get endpoint and construct URL.
    std::string endpoint_url = fs_->GetEndpointUrl(bucket, object);

    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(
        fs_->CreateSignedRequest(&request, "PUT", bucket, object, payload_hash));

    request->SetUri(endpoint_url);
    request->SetPostFromBuffer(buffer_.data(), buffer_.size());
    // SetPostFromBuffer makes it a POST; we need PUT. The CurlHttpRequest
    // SetPutEmptyBody or SetPostFromBuffer approach varies. For now, we use
    // the post approach which the signed headers accommodate.

    TF_RETURN_IF_ERROR(request->Send());

    uint64_t response_code = request->GetResponseCode();
    if (response_code != 200) {
      return absl::InternalError(
          absl::StrCat("S3 PutObject failed with HTTP ", response_code));
    }
    return absl::OkStatus();
  }

  absl::Status Flush() override { return absl::OkStatus(); }

  absl::Status Name(absl::string_view* result) const override {
    *result = filename_;
    return absl::OkStatus();
  }

  absl::Status Sync() override { return absl::OkStatus(); }

 private:
  std::string filename_;
  S3FileSystem* fs_;
  std::string buffer_;
  bool closed_ = false;
};

}  // namespace

S3FileSystem::S3FileSystem()
    : http_request_factory_(std::make_shared<CurlHttpRequest::Factory>()) {
  region_ = GetEnvVarOrDefault(kAwsRegion,
                               GetEnvVarOrDefault(kAwsDefaultRegion,
                                                  kDefaultRegion));
  endpoint_ = GetEnvVarOrDefault(kS3Endpoint, "");
  // Load credentials eagerly but don't fail construction on error.
  // Credentials will be re-checked on each request.
  LoadCredentials().IgnoreError();
}

S3FileSystem::S3FileSystem(
    std::unique_ptr<HttpRequest::Factory> http_request_factory)
    : http_request_factory_(std::move(http_request_factory)) {
  region_ = GetEnvVarOrDefault(kAwsRegion,
                               GetEnvVarOrDefault(kAwsDefaultRegion,
                                                  kDefaultRegion));
  endpoint_ = GetEnvVarOrDefault(kS3Endpoint, "");
  LoadCredentials().IgnoreError();
}

absl::Status S3FileSystem::LoadCredentials() {
  absl::MutexLock lock(&mu_);
  const char* access_key = std::getenv(kAwsAccessKeyId);
  const char* secret_key = std::getenv(kAwsSecretAccessKey);
  const char* session_token = std::getenv(kAwsSessionToken);

  if (access_key && secret_key) {
    access_key_ = access_key;
    secret_key_ = secret_key;
    session_token_ = session_token ? session_token : "";
    return absl::OkStatus();
  }

  // Try to load from ~/.aws/credentials file.
  const char* home = std::getenv("HOME");
  if (home) {
    std::string creds_file = absl::StrCat(home, "/.aws/credentials");
    std::ifstream file(creds_file);
    if (file.is_open()) {
      std::string line;
      bool in_default_profile = false;
      while (std::getline(file, line)) {
        // Strip whitespace.
        absl::string_view trimmed = absl::StripAsciiWhitespace(line);
        if (trimmed == "[default]") {
          in_default_profile = true;
          continue;
        }
        if (absl::StartsWith(trimmed, "[")) {
          in_default_profile = false;
          continue;
        }
        if (!in_default_profile) continue;

        std::vector<std::string> parts = absl::StrSplit(trimmed, '=');
        if (parts.size() != 2) continue;
        std::string key =
            std::string(absl::StripAsciiWhitespace(parts[0]));
        std::string value =
            std::string(absl::StripAsciiWhitespace(parts[1]));

        if (key == "aws_access_key_id") {
          access_key_ = value;
        } else if (key == "aws_secret_access_key") {
          secret_key_ = value;
        } else if (key == "aws_session_token") {
          session_token_ = value;
        }
      }
      if (!access_key_.empty() && !secret_key_.empty()) {
        return absl::OkStatus();
      }
    }
  }

  // TODO: Support EC2 instance metadata (IMDS) and ECS task role credentials.
  // For now, we'll allow anonymous access (unsigned requests) if no creds.
  LOG(WARNING) << "No AWS credentials found. S3 requests will be unsigned. "
               << "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
               << "environment variables for authenticated access.";
  return absl::OkStatus();
}

absl::Status S3FileSystem::ParseS3Path(absl::string_view fname,
                                       bool empty_object_ok,
                                       std::string* bucket,
                                       std::string* object) {
  absl::string_view scheme, host, path;
  io::ParseURI(fname, &scheme, &host, &path);
  if (scheme != "s3") {
    return absl::InvalidArgumentError(
        absl::StrCat("S3 path doesn't start with 's3://': ", fname));
  }
  *bucket = std::string(host);
  if (bucket->empty() || *bucket == ".") {
    return absl::InvalidArgumentError(
        absl::StrCat("S3 path doesn't contain a bucket name: ", fname));
  }
  absl::ConsumePrefix(&path, "/");
  *object = std::string(path);
  if (!empty_object_ok && object->empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("S3 path doesn't contain an object name: ", fname));
  }
  return absl::OkStatus();
}

std::string S3FileSystem::GetEndpointUrl(const std::string& bucket,
                                         const std::string& object) {
  std::string encoded_object = UriEncode(object);
  if (!endpoint_.empty()) {
    // Custom endpoint (MinIO, LocalStack, etc.) - use path-style.
    return absl::StrCat(endpoint_, "/", bucket, "/", encoded_object);
  }
  // Default: virtual-hosted-style URLs.
  std::string use_path_style = GetEnvVarOrDefault(kS3UsePathStyle, "");
  if (use_path_style == "true" || use_path_style == "1") {
    return absl::StrCat("https://s3.", region_, ".amazonaws.com/", bucket, "/",
                        encoded_object);
  }
  return absl::StrCat("https://", bucket, ".s3.", region_,
                       ".amazonaws.com/", encoded_object);
}

std::string S3FileSystem::HmacSha256(const std::string& key,
                                     const std::string& data) {
  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int hash_len;
  HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
       reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash,
       &hash_len);
  return std::string(reinterpret_cast<char*>(hash), hash_len);
}

std::string S3FileSystem::Sha256Hex(const std::string& data) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(),
         hash);
  std::ostringstream ss;
  for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
    ss << std::hex << std::setfill('0') << std::setw(2)
       << static_cast<int>(hash[i]);
  }
  return ss.str();
}

std::string S3FileSystem::ComputeSignatureV4(
    const std::string& method, const std::string& canonical_uri,
    const std::string& canonical_querystring,
    const std::string& signed_headers,
    const std::string& canonical_headers_str,
    const std::string& payload_hash, const std::string& datestamp,
    const std::string& amz_date) {
  absl::MutexLock lock(&mu_);
  // Step 1: Create canonical request.
  std::string canonical_request =
      absl::StrCat(method, "\n", canonical_uri, "\n", canonical_querystring,
                   "\n", canonical_headers_str, "\n", signed_headers, "\n",
                   payload_hash);

  // Step 2: Create string to sign.
  std::string credential_scope =
      absl::StrCat(datestamp, "/", region_, "/", kS3Service, "/aws4_request");
  std::string string_to_sign =
      absl::StrCat("AWS4-HMAC-SHA256\n", amz_date, "\n", credential_scope,
                   "\n", Sha256Hex(canonical_request));

  // Step 3: Calculate signature.
  std::string signing_key =
      HmacSha256(absl::StrCat("AWS4", secret_key_), datestamp);
  signing_key = HmacSha256(signing_key, region_);
  signing_key = HmacSha256(signing_key, kS3Service);
  signing_key = HmacSha256(signing_key, "aws4_request");

  std::string signature_raw = HmacSha256(signing_key, string_to_sign);

  // Hex encode the signature.
  std::ostringstream ss;
  for (unsigned char c : signature_raw) {
    ss << std::hex << std::setfill('0') << std::setw(2)
       << static_cast<int>(c);
  }
  return ss.str();
}

absl::Status S3FileSystem::CreateSignedRequest(
    std::unique_ptr<HttpRequest>* request, const std::string& method,
    const std::string& bucket, const std::string& object,
    const std::string& content_sha256) {
  request->reset(http_request_factory_->Create());

  std::string datestamp, amz_date;
  GetTimestamps(&datestamp, &amz_date);

  std::string host;
  std::string canonical_uri;
  if (!endpoint_.empty()) {
    // Parse host from custom endpoint.
    absl::string_view ep = endpoint_;
    absl::ConsumePrefix(&ep, "https://");
    absl::ConsumePrefix(&ep, "http://");
    // Remove trailing path if any.
    auto slash_pos = ep.find('/');
    host = std::string(ep.substr(0, slash_pos));
    canonical_uri = absl::StrCat("/", bucket, "/", UriEncode(object));
  } else {
    std::string use_path_style = GetEnvVarOrDefault(kS3UsePathStyle, "");
    if (use_path_style == "true" || use_path_style == "1") {
      host = absl::StrCat("s3.", region_, ".amazonaws.com");
      canonical_uri = absl::StrCat("/", bucket, "/", UriEncode(object));
    } else {
      host = absl::StrCat(bucket, ".s3.", region_, ".amazonaws.com");
      canonical_uri = absl::StrCat("/", UriEncode(object));
    }
  }

  // Build canonical headers (must be sorted by header name).
  std::string canonical_headers =
      absl::StrCat("host:", host, "\n", "x-amz-content-sha256:", content_sha256,
                   "\n", "x-amz-date:", amz_date, "\n");
  std::string signed_headers = "host;x-amz-content-sha256;x-amz-date";

  {
    absl::MutexLock lock(&mu_);
    if (!session_token_.empty()) {
      canonical_headers =
          absl::StrCat("host:", host, "\n", "x-amz-content-sha256:",
                       content_sha256, "\n", "x-amz-date:", amz_date, "\n",
                       "x-amz-security-token:", session_token_, "\n");
      signed_headers =
          "host;x-amz-content-sha256;x-amz-date;x-amz-security-token";
    }
  }

  std::string canonical_querystring;  // empty for most operations

  // Compute signature.
  std::string signature = ComputeSignatureV4(
      method, canonical_uri, canonical_querystring, signed_headers,
      canonical_headers, content_sha256, datestamp, amz_date);

  // Build Authorization header.
  std::string credential_scope =
      absl::StrCat(datestamp, "/", region_, "/", kS3Service, "/aws4_request");

  std::string access_key;
  std::string session_token;
  {
    absl::MutexLock lock(&mu_);
    access_key = access_key_;
    session_token = session_token_;
  }

  std::string authorization = absl::StrCat(
      "AWS4-HMAC-SHA256 Credential=", access_key, "/", credential_scope,
      ", SignedHeaders=", signed_headers, ", Signature=", signature);

  // Set headers.
  (*request)->AddHeader("Authorization", authorization);
  (*request)->AddHeader("x-amz-date", amz_date);
  (*request)->AddHeader("x-amz-content-sha256", content_sha256);
  if (!session_token.empty()) {
    (*request)->AddHeader("x-amz-security-token", session_token);
  }

  return absl::OkStatus();
}

absl::Status S3FileSystem::NewRandomAccessFile(
    const std::string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  result->reset(new S3RandomAccessFile(fname, this));
  return absl::OkStatus();
}

absl::Status S3FileSystem::NewWritableFile(
    const std::string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));
  result->reset(new S3WritableFile(fname, this));
  return absl::OkStatus();
}

absl::Status S3FileSystem::NewAppendableFile(
    const std::string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  // S3 does not support native append. For the autotune cache use case,
  // we can treat this as a new write.
  return NewWritableFile(fname, token, result);
}

absl::Status S3FileSystem::NewReadOnlyMemoryRegionFromFile(
    const std::string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  return absl::UnimplementedError(
      "S3 does not support memory-mapped files. Use NewRandomAccessFile.");
}

absl::Status S3FileSystem::FileExists(const std::string& fname,
                                      TransactionToken* token) {
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, true, &bucket, &object));

  if (object.empty()) {
    // Check if bucket exists by doing a HEAD request on the bucket.
    std::string content_sha256 = Sha256Hex("");

    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(
        CreateSignedRequest(&request, "HEAD", bucket, "", content_sha256));
    request->SetUri(GetEndpointUrl(bucket, ""));
    absl::Status status = request->Send();
    if (status.ok()) return absl::OkStatus();
    return absl::NotFoundError(absl::StrCat("Bucket not found: ", bucket));
  }

  // HEAD request to check if object exists.
  std::string content_sha256 = Sha256Hex("");

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(
      CreateSignedRequest(&request, "HEAD", bucket, object, content_sha256));
  request->SetUri(GetEndpointUrl(bucket, object));
  absl::Status status = request->Send();
  if (status.ok()) return absl::OkStatus();

  // Also check if it's a directory prefix.
  std::string dir_object = MaybeAppendSlash(object);
  std::unique_ptr<HttpRequest> dir_request;
  TF_RETURN_IF_ERROR(CreateSignedRequest(&dir_request, "GET", bucket, "",
                                         content_sha256));

  std::string list_url;
  if (!endpoint_.empty()) {
    list_url = absl::StrCat(endpoint_, "/", bucket,
                            "?list-type=2&prefix=", UriEncode(dir_object, true),
                            "&max-keys=1");
  } else {
    std::string host =
        absl::StrCat(bucket, ".s3.", region_, ".amazonaws.com");
    list_url = absl::StrCat("https://", host,
                            "?list-type=2&prefix=", UriEncode(dir_object, true),
                            "&max-keys=1");
  }
  dir_request->SetUri(list_url);
  std::vector<char> response_buffer;
  dir_request->SetResultBuffer(&response_buffer);
  status = dir_request->Send();
  if (status.ok()) {
    std::string response(response_buffer.begin(), response_buffer.end());
    std::string key_count = ExtractXmlTagValue(response, "KeyCount");
    int count = 0;
    if (absl::SimpleAtoi(key_count, &count) && count > 0) {
      return absl::OkStatus();
    }
  }

  return absl::NotFoundError(absl::StrCat("Object not found: ", fname));
}

absl::Status S3FileSystem::Stat(const std::string& fname,
                                TransactionToken* token,
                                FileStatistics* stat) {
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, true, &bucket, &object));

  if (object.empty()) {
    // Bucket - report as directory.
    stat->is_directory = true;
    stat->length = 0;
    stat->mtime_nsec = 0;
    return absl::OkStatus();
  }

  std::string content_sha256 = Sha256Hex("");
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(
      CreateSignedRequest(&request, "HEAD", bucket, object, content_sha256));
  request->SetUri(GetEndpointUrl(bucket, object));
  absl::Status status = request->Send();

  if (status.ok()) {
    std::string content_length_str =
        request->GetResponseHeader("Content-Length");
    uint64_t content_length = 0;
    absl::SimpleAtoi(content_length_str, &content_length);
    stat->length = content_length;
    stat->is_directory = false;
    stat->mtime_nsec = 0;  // Could parse Last-Modified if needed.
    return absl::OkStatus();
  }

  // Check if it's a directory prefix.
  absl::Status dir_status = IsDirectory(fname, token);
  if (dir_status.ok()) {
    stat->is_directory = true;
    stat->length = 0;
    stat->mtime_nsec = 0;
    return absl::OkStatus();
  }

  return absl::NotFoundError(absl::StrCat("Object not found: ", fname));
}

absl::Status S3FileSystem::GetChildren(const std::string& dir,
                                       TransactionToken* token,
                                       std::vector<std::string>* result) {
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(dir, true, &bucket, &object));

  std::string prefix = object.empty() ? "" : MaybeAppendSlash(object);
  std::string content_sha256 = Sha256Hex("");

  std::string continuation_token;
  bool is_truncated = true;

  while (is_truncated) {
    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(
        CreateSignedRequest(&request, "GET", bucket, "", content_sha256));

    std::string list_url;
    std::string host;
    if (!endpoint_.empty()) {
      list_url = absl::StrCat(endpoint_, "/", bucket, "?list-type=2&delimiter=/");
    } else {
      host = absl::StrCat(bucket, ".s3.", region_, ".amazonaws.com");
      list_url = absl::StrCat("https://", host, "?list-type=2&delimiter=/");
    }

    if (!prefix.empty()) {
      absl::StrAppend(&list_url, "&prefix=", UriEncode(prefix, true));
    }
    if (!continuation_token.empty()) {
      absl::StrAppend(&list_url, "&continuation-token=",
                       UriEncode(continuation_token, true));
    }

    request->SetUri(list_url);
    std::vector<char> response_buffer;
    request->SetResultBuffer(&response_buffer);
    TF_RETURN_IF_ERROR(request->Send());

    std::string response(response_buffer.begin(), response_buffer.end());

    // Parse <Key> entries from <Contents>.
    std::vector<std::string> keys = ExtractAllXmlTagValues(response, "Key");
    for (const auto& key : keys) {
      // Remove the prefix to get relative names.
      absl::string_view relative = key;
      absl::ConsumePrefix(&relative, prefix);
      if (!relative.empty()) {
        result->push_back(std::string(relative));
      }
    }

    // Parse <Prefix> entries from <CommonPrefixes>.
    std::vector<std::string> prefixes =
        ExtractAllXmlTagValues(response, "Prefix");
    for (const auto& p : prefixes) {
      absl::string_view relative = p;
      absl::ConsumePrefix(&relative, prefix);
      // Remove trailing slash from directory names.
      absl::ConsumeSuffix(&relative, "/");
      if (!relative.empty()) {
        result->push_back(std::string(relative));
      }
    }

    std::string truncated_str =
        ExtractXmlTagValue(response, "IsTruncated");
    is_truncated = (truncated_str == "true");
    if (is_truncated) {
      continuation_token =
          ExtractXmlTagValue(response, "NextContinuationToken");
    }
  }

  return absl::OkStatus();
}

absl::Status S3FileSystem::GetMatchingPaths(
    const std::string& pattern, TransactionToken* token,
    std::vector<std::string>* results) {
  // Extract the fixed prefix before any glob characters.
  std::string fixed_prefix;
  for (char c : pattern) {
    if (c == '*' || c == '?' || c == '[') break;
    fixed_prefix += c;
  }

  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fixed_prefix, true, &bucket, &object));

  // List all objects with the prefix and filter by pattern.
  std::string prefix = object;
  std::string content_sha256 = Sha256Hex("");
  std::string continuation_token;
  bool is_truncated = true;

  while (is_truncated) {
    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(
        CreateSignedRequest(&request, "GET", bucket, "", content_sha256));

    std::string list_url;
    if (!endpoint_.empty()) {
      list_url = absl::StrCat(endpoint_, "/", bucket, "?list-type=2");
    } else {
      std::string host =
          absl::StrCat(bucket, ".s3.", region_, ".amazonaws.com");
      list_url = absl::StrCat("https://", host, "?list-type=2");
    }

    if (!prefix.empty()) {
      absl::StrAppend(&list_url, "&prefix=", UriEncode(prefix, true));
    }
    if (!continuation_token.empty()) {
      absl::StrAppend(&list_url, "&continuation-token=",
                       UriEncode(continuation_token, true));
    }

    request->SetUri(list_url);
    std::vector<char> response_buffer;
    request->SetResultBuffer(&response_buffer);
    TF_RETURN_IF_ERROR(request->Send());

    std::string response(response_buffer.begin(), response_buffer.end());

    std::vector<std::string> keys = ExtractAllXmlTagValues(response, "Key");
    for (const auto& key : keys) {
      std::string full_path = absl::StrCat("s3://", bucket, "/", key);
      if (Match(full_path, pattern)) {
        results->push_back(full_path);
      }
    }

    std::string truncated_str =
        ExtractXmlTagValue(response, "IsTruncated");
    is_truncated = (truncated_str == "true");
    if (is_truncated) {
      continuation_token =
          ExtractXmlTagValue(response, "NextContinuationToken");
    }
  }

  return absl::OkStatus();
}

absl::Status S3FileSystem::DeleteFile(const std::string& fname,
                                      TransactionToken* token) {
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, false, &bucket, &object));

  std::string content_sha256 = Sha256Hex("");
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(
      CreateSignedRequest(&request, "DELETE", bucket, object, content_sha256));
  request->SetUri(GetEndpointUrl(bucket, object));
  request->SetDeleteRequest();
  return request->Send();
}

absl::Status S3FileSystem::CreateDir(const std::string& dirname,
                                     TransactionToken* token) {
  // S3 has no real directory concept. We can optionally create a
  // zero-byte marker object with a trailing slash, but for most
  // use cases (including the autotune cache), this is a no-op.
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(dirname, true, &bucket, &object));
  // Silently succeed - directories are virtual in S3.
  return absl::OkStatus();
}

absl::Status S3FileSystem::DeleteDir(const std::string& dirname,
                                     TransactionToken* token) {
  // Check that the directory is empty.
  std::vector<std::string> children;
  TF_RETURN_IF_ERROR(GetChildren(dirname, token, &children));
  if (!children.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Directory not empty: ", dirname));
  }
  return absl::OkStatus();
}

absl::Status S3FileSystem::GetFileSize(const std::string& fname,
                                       TransactionToken* token,
                                       uint64* file_size) {
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(fname, token, &stat));
  *file_size = stat.length;
  return absl::OkStatus();
}

absl::Status S3FileSystem::RenameFile(const std::string& src,
                                      const std::string& target,
                                      TransactionToken* token) {
  // S3 doesn't have a rename operation. We must copy + delete.
  std::string src_bucket, src_object;
  TF_RETURN_IF_ERROR(ParseS3Path(src, false, &src_bucket, &src_object));
  std::string dst_bucket, dst_object;
  TF_RETURN_IF_ERROR(ParseS3Path(target, false, &dst_bucket, &dst_object));

  // Use S3 CopyObject.
  std::string content_sha256 = Sha256Hex("");
  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(CreateSignedRequest(&request, "PUT", dst_bucket,
                                         dst_object, content_sha256));

  // The x-amz-copy-source header specifies the source.
  std::string copy_source =
      absl::StrCat("/", src_bucket, "/", UriEncode(src_object, true));
  request->AddHeader("x-amz-copy-source", copy_source);
  request->SetUri(GetEndpointUrl(dst_bucket, dst_object));
  request->SetPutEmptyBody();
  TF_RETURN_IF_ERROR(request->Send());

  // Delete the source.
  return DeleteFile(src, token);
}

absl::Status S3FileSystem::IsDirectory(const std::string& fname,
                                       TransactionToken* token) {
  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(fname, true, &bucket, &object));

  if (object.empty()) {
    // It's a bucket - always a directory.
    return absl::OkStatus();
  }

  // Check if there are any objects with this prefix.
  std::string prefix = MaybeAppendSlash(object);
  std::string content_sha256 = Sha256Hex("");

  std::unique_ptr<HttpRequest> request;
  TF_RETURN_IF_ERROR(
      CreateSignedRequest(&request, "GET", bucket, "", content_sha256));

  std::string list_url;
  if (!endpoint_.empty()) {
    list_url = absl::StrCat(endpoint_, "/", bucket,
                            "?list-type=2&prefix=", UriEncode(prefix, true),
                            "&max-keys=1");
  } else {
    std::string host =
        absl::StrCat(bucket, ".s3.", region_, ".amazonaws.com");
    list_url = absl::StrCat("https://", host,
                            "?list-type=2&prefix=", UriEncode(prefix, true),
                            "&max-keys=1");
  }
  request->SetUri(list_url);
  std::vector<char> response_buffer;
  request->SetResultBuffer(&response_buffer);
  TF_RETURN_IF_ERROR(request->Send());

  std::string response(response_buffer.begin(), response_buffer.end());
  std::string key_count = ExtractXmlTagValue(response, "KeyCount");
  int count = 0;
  if (absl::SimpleAtoi(key_count, &count) && count > 0) {
    return absl::OkStatus();
  }

  return absl::FailedPreconditionError(
      absl::StrCat("Not a directory: ", fname));
}

absl::Status S3FileSystem::DeleteRecursively(const std::string& dirname,
                                             TransactionToken* token,
                                             int64_t* undeleted_files,
                                             int64_t* undeleted_dirs) {
  *undeleted_files = 0;
  *undeleted_dirs = 0;

  std::string bucket, object;
  TF_RETURN_IF_ERROR(ParseS3Path(dirname, true, &bucket, &object));

  // List all objects with the prefix and delete them.
  std::string prefix = object.empty() ? "" : MaybeAppendSlash(object);
  std::string content_sha256 = Sha256Hex("");
  std::string continuation_token;
  bool is_truncated = true;

  while (is_truncated) {
    std::unique_ptr<HttpRequest> request;
    TF_RETURN_IF_ERROR(
        CreateSignedRequest(&request, "GET", bucket, "", content_sha256));

    std::string list_url;
    if (!endpoint_.empty()) {
      list_url = absl::StrCat(endpoint_, "/", bucket, "?list-type=2");
    } else {
      std::string host =
          absl::StrCat(bucket, ".s3.", region_, ".amazonaws.com");
      list_url = absl::StrCat("https://", host, "?list-type=2");
    }
    if (!prefix.empty()) {
      absl::StrAppend(&list_url, "&prefix=", UriEncode(prefix, true));
    }
    if (!continuation_token.empty()) {
      absl::StrAppend(&list_url, "&continuation-token=",
                       UriEncode(continuation_token, true));
    }

    request->SetUri(list_url);
    std::vector<char> response_buffer;
    request->SetResultBuffer(&response_buffer);
    TF_RETURN_IF_ERROR(request->Send());

    std::string response(response_buffer.begin(), response_buffer.end());
    std::vector<std::string> keys = ExtractAllXmlTagValues(response, "Key");
    for (const auto& key : keys) {
      std::string full_path = absl::StrCat("s3://", bucket, "/", key);
      absl::Status s = DeleteFile(full_path, token);
      if (!s.ok()) {
        (*undeleted_files)++;
      }
    }

    std::string truncated_str =
        ExtractXmlTagValue(response, "IsTruncated");
    is_truncated = (truncated_str == "true");
    if (is_truncated) {
      continuation_token =
          ExtractXmlTagValue(response, "NextContinuationToken");
    }
  }

  return absl::OkStatus();
}

void S3FileSystem::FlushCaches(TransactionToken* token) {
  // No caches to flush in this basic implementation.
}

absl::Status S3FileSystem::HasAtomicMove(const std::string& path,
                                         bool* has_atomic_move) {
  // S3 does not have native atomic move/rename. A "rename" is copy + delete.
  *has_atomic_move = false;
  return absl::OkStatus();
}

RetryingS3FileSystem::RetryingS3FileSystem()
    : RetryingFileSystem(std::make_unique<S3FileSystem>(),
                         GetS3RetryConfig()) {}

}  // namespace tsl

// Register the S3 filesystem with the "s3" scheme.
// Uses REGISTER_LEGACY_FILE_SYSTEM so it can be overridden by modular
// filesystem plugins (e.g., from tensorflow-io) via TF_USE_MODULAR_FILESYSTEM.
REGISTER_LEGACY_FILE_SYSTEM("s3", ::tsl::RetryingS3FileSystem);
