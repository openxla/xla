#!/usr/bin/env python3
"""Integration test for S3 filesystem support in XLA/TSL.

This test exercises the S3 filesystem through the tsl::Env layer by driving
JAX/XLA operations that use the per-fusion autotune cache on S3.

Usage:
  # Against real AWS S3:
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
    python s3_file_system_integration_test.py --bucket my-test-bucket

  # Against LocalStack (start with: docker run -p 4566:4566 localstack/localstack):
  S3_ENDPOINT=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test \
    python s3_file_system_integration_test.py --bucket test-bucket --create-bucket

  # Against MinIO (start with: docker run -p 9000:9000 minio/minio server /data):
  S3_ENDPOINT=http://localhost:9000 AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
    python s3_file_system_integration_test.py --bucket test-bucket --create-bucket
"""

import argparse
import os
import sys
import tempfile
import time
import uuid

# ---------------------------------------------------------------------------
# Helpers that only need boto3
# ---------------------------------------------------------------------------

def get_s3_client(endpoint_url=None):
    """Create a boto3 S3 client, optionally pointing at a custom endpoint."""
    import boto3
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)


def ensure_bucket(s3, bucket, create=False):
    """Check that the bucket exists, optionally creating it."""
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  Bucket s3://{bucket} exists.")
    except Exception:
        if create:
            s3.create_bucket(Bucket=bucket)
            print(f"  Created bucket s3://{bucket}.")
        else:
            print(f"  ERROR: Bucket s3://{bucket} does not exist. "
                  f"Pass --create-bucket to create it.")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class S3FileSystemTests:
    """Tests that exercise S3 filesystem operations used by XLA's autotune cache.

    These mirror exactly the operations performed by:
      - xla/service/gpu/autotuning/autotuner_util.cc
      - xla/backends/autotuner/file_based_autotuner_cache.cc
    """

    def __init__(self, s3, bucket, prefix, endpoint_url=None):
        self.s3 = s3
        self.bucket = bucket
        self.prefix = prefix
        self.endpoint_url = endpoint_url
        self.passed = 0
        self.failed = 0

    def _s3_uri(self, key):
        return f"s3://{self.bucket}/{self.prefix}/{key}"

    def _check(self, name, condition, detail=""):
        if condition:
            print(f"  PASS: {name}")
            self.passed += 1
        else:
            print(f"  FAIL: {name} {detail}")
            self.failed += 1

    # -- boto3-level tests (verify S3 itself works) --

    def test_put_and_get_object(self):
        """Write a file and read it back via boto3."""
        key = f"{self.prefix}/test_put_get.txt"
        body = f"hello-{uuid.uuid4()}"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body.encode())
        resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        got = resp["Body"].read().decode()
        self._check("put_and_get_object", got == body,
                    f"expected {body!r}, got {got!r}")

    def test_head_object(self):
        """Check that HEAD returns metadata for an existing object."""
        key = f"{self.prefix}/test_head.txt"
        content = b"some content for head test"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=content)
        resp = self.s3.head_object(Bucket=self.bucket, Key=key)
        self._check("head_object_exists",
                    resp["ContentLength"] == len(content))

    def test_head_object_missing(self):
        """HEAD on a nonexistent key should raise an error."""
        key = f"{self.prefix}/does_not_exist_{uuid.uuid4()}.txt"
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            self._check("head_object_missing", False, "expected 404")
        except self.s3.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            self._check("head_object_missing", code == "404",
                        f"expected 404, got {code}")

    def test_list_objects(self):
        """List objects with a prefix (used by GetMatchingPaths)."""
        base = f"{self.prefix}/list_test"
        for i in range(3):
            self.s3.put_object(Bucket=self.bucket,
                             Key=f"{base}/file{i}.textproto",
                             Body=f"entry {i}".encode())
        resp = self.s3.list_objects_v2(Bucket=self.bucket,
                                       Prefix=f"{base}/",
                                       MaxKeys=100)
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        self._check("list_objects",
                    len(keys) == 3 and all(".textproto" in k for k in keys),
                    f"got {keys}")

    def test_delete_object(self):
        """Delete an object (used when overwriting cache entries)."""
        key = f"{self.prefix}/test_delete.txt"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=b"delete me")
        self.s3.delete_object(Bucket=self.bucket, Key=key)
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            self._check("delete_object", False, "object still exists")
        except self.s3.exceptions.ClientError:
            self._check("delete_object", True)

    def test_copy_object(self):
        """Copy an object (used by RenameFile = copy + delete)."""
        src_key = f"{self.prefix}/copy_src.txt"
        dst_key = f"{self.prefix}/copy_dst.txt"
        body = b"copy test content"
        self.s3.put_object(Bucket=self.bucket, Key=src_key, Body=body)
        self.s3.copy_object(
            Bucket=self.bucket, Key=dst_key,
            CopySource={"Bucket": self.bucket, "Key": src_key})
        resp = self.s3.get_object(Bucket=self.bucket, Key=dst_key)
        got = resp["Body"].read()
        self._check("copy_object", got == body,
                    f"expected {body!r}, got {got!r}")

    # -- XLA autotune cache simulation --

    def test_autotune_cache_workflow(self):
        """Simulate the exact workflow of XLA's per-fusion autotune cache.

        The autotune cache does:
          1. GetMatchingPaths("s3://bucket/cache_dir/*.textproto")
          2. ReadFileToString for each match
          3. RecursivelyCreateDir (no-op on S3)
          4. WriteStringToFile to a temp path
          5. RenameFile (copy + delete) from temp to final
        """
        cache_dir = f"{self.prefix}/autotune_cache"
        tmp_dir = f"{cache_dir}/tmp"

        # Step 1: Initially empty.
        resp = self.s3.list_objects_v2(
            Bucket=self.bucket, Prefix=f"{cache_dir}/", MaxKeys=100)
        initial_count = resp.get("KeyCount", 0)

        # Step 3: RecursivelyCreateDir is a no-op for S3 (dirs are virtual).
        # Just verify we can proceed.

        # Step 4: Write to temp location.
        textproto_content = (
            'key {\n'
            '  hlo_fingerprint: "abc123"\n'
            '  device_str: "CUDA: 8.0"\n'
            '  version: 1\n'
            '}\n'
            'codegen_backend: "TRITON"\n'
        )
        temp_key = f"{tmp_dir}/tmp_cache_abc123_{int(time.time())}.textproto"
        self.s3.put_object(Bucket=self.bucket, Key=temp_key,
                          Body=textproto_content.encode())

        # Step 5: Rename (copy + delete) to final location.
        final_key = f"{cache_dir}/abc123.textproto"
        self.s3.copy_object(
            Bucket=self.bucket, Key=final_key,
            CopySource={"Bucket": self.bucket, "Key": temp_key})
        self.s3.delete_object(Bucket=self.bucket, Key=temp_key)

        # Verify temp is gone and final exists.
        try:
            self.s3.head_object(Bucket=self.bucket, Key=temp_key)
            self._check("autotune_rename_temp_deleted", False)
        except self.s3.exceptions.ClientError:
            self._check("autotune_rename_temp_deleted", True)

        resp = self.s3.get_object(Bucket=self.bucket, Key=final_key)
        got = resp["Body"].read().decode()
        self._check("autotune_rename_final_exists",
                    got == textproto_content)

        # Step 1 again: GetMatchingPaths should find our entry.
        resp = self.s3.list_objects_v2(
            Bucket=self.bucket, Prefix=f"{cache_dir}/", MaxKeys=100)
        textproto_keys = [
            obj["Key"] for obj in resp.get("Contents", [])
            if obj["Key"].endswith(".textproto") and "/tmp/" not in obj["Key"]
        ]
        self._check("autotune_list_finds_entry",
                    any("abc123.textproto" in k for k in textproto_keys),
                    f"got {textproto_keys}")

        # Step 2: ReadFileToString.
        for key in textproto_keys:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            content = resp["Body"].read().decode()
            self._check(f"autotune_read_{os.path.basename(key)}",
                        len(content) > 0)

    # -- Optional: JAX integration test --

    def test_jax_autotune_cache_dir(self):
        """If JAX is available, verify XLA accepts an S3 cache dir.

        This test configures xla_gpu_per_fusion_autotune_cache_dir to an
        S3 path and runs a trivial computation to trigger the cache code path.
        It may fail if XLA was not built with S3 filesystem support (which is
        exactly what this PR adds).
        """
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            print("  SKIP: test_jax_autotune_cache_dir (jax not installed)")
            return

        cache_uri = f"s3://{self.bucket}/{self.prefix}/jax_cache"

        # Set the XLA flag for the autotune cache dir.
        os.environ["XLA_FLAGS"] = (
            f"--xla_gpu_per_fusion_autotune_cache_dir={cache_uri}"
        )
        if self.endpoint_url:
            os.environ["S3_ENDPOINT"] = self.endpoint_url

        try:
            # A simple matmul will trigger autotuning on GPU.
            a = jnp.ones((32, 32))
            b = jnp.ones((32, 32))
            result = jnp.dot(a, b)
            result.block_until_ready()
            self._check("jax_autotune_cache_dir_no_crash", True)
        except Exception as e:
            error_str = str(e)
            if "File system scheme 's3' not implemented" in error_str:
                self._check("jax_autotune_cache_dir_no_crash", False,
                            "S3 filesystem not registered in XLA. "
                            "This is the bug this PR fixes.")
            else:
                self._check("jax_autotune_cache_dir_no_crash", False,
                            f"unexpected error: {e}")
        finally:
            # Clean up env.
            if "XLA_FLAGS" in os.environ:
                del os.environ["XLA_FLAGS"]

    def run_all(self):
        """Run all test methods."""
        tests = [m for m in dir(self) if m.startswith("test_")]
        print(f"\nRunning {len(tests)} tests against "
              f"s3://{self.bucket}/{self.prefix}\n")
        for test_name in sorted(tests):
            try:
                getattr(self, test_name)()
            except Exception as e:
                print(f"  FAIL: {test_name} raised {type(e).__name__}: {e}")
                self.failed += 1
        return self.passed, self.failed

    def cleanup(self):
        """Remove all test objects from S3."""
        print(f"\nCleaning up s3://{self.bucket}/{self.prefix}/ ...")
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket,
                                        Prefix=f"{self.prefix}/"):
            objects = [{"Key": obj["Key"]}
                       for obj in page.get("Contents", [])]
            if objects:
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects})
        print("  Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Integration test for XLA S3 filesystem support")
    parser.add_argument("--bucket", required=True,
                        help="S3 bucket to use for testing")
    parser.add_argument("--prefix", default=None,
                        help="Key prefix for test objects "
                             "(default: auto-generated)")
    parser.add_argument("--create-bucket", action="store_true",
                        help="Create the bucket if it doesn't exist")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't delete test objects after running")
    parser.add_argument("--skip-jax", action="store_true",
                        help="Skip JAX integration test")
    args = parser.parse_args()

    endpoint_url = os.environ.get("S3_ENDPOINT")
    prefix = args.prefix or f"xla-s3-test-{uuid.uuid4().hex[:8]}"

    print("S3 Filesystem Integration Test")
    print("=" * 50)
    print(f"  Endpoint: {endpoint_url or 'AWS default'}")
    print(f"  Bucket:   {args.bucket}")
    print(f"  Prefix:   {prefix}")
    print()

    s3 = get_s3_client(endpoint_url)
    ensure_bucket(s3, args.bucket, create=args.create_bucket)

    tests = S3FileSystemTests(s3, args.bucket, prefix, endpoint_url)

    if args.skip_jax:
        # Monkey-patch to skip the JAX test.
        tests.test_jax_autotune_cache_dir = lambda: print(
            "  SKIP: test_jax_autotune_cache_dir (--skip-jax)")

    try:
        passed, failed = tests.run_all()
    finally:
        if not args.no_cleanup:
            tests.cleanup()

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
