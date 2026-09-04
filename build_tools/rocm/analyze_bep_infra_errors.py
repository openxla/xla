#!/usr/bin/env python3
# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check BEP JSON for infrastructure error events."""

import json
import sys

if len(sys.argv) != 2:
    sys.exit(2)

bep_file = sys.argv[1]

try:
    with open(bep_file, "r") as f:
        events = [json.loads(line) for line in f if line.strip()]
except:
    sys.exit(0)

# Infrastructure failure reasons
INFRA_REASONS = [
    "REMOTE_FAILURE",
    "OUT_OF_MEMORY",
    "INTERNAL",
    "LOADING_FAILURE",
    "NO_ANALYZE",
    "NO_BUILD",
]

found = False

for e in events:
    if "aborted" in e:
        reason = e["aborted"].get("reason", "")
        if reason in INFRA_REASONS:
            print(f"aborted.reason={reason}")  # DISABLE_DEBUG_PRINT_CHECK
            found = True

sys.exit(1 if found else 0)
