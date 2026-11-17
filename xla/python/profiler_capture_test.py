# Copyright 2026 The OpenXLA Authors.
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
# ==============================================================================

from absl.testing import absltest
from xla.python import _profiler


class ProfilerCaptureTest(absltest.TestCase):

  def test_capture_without_cuda_context_raises(self):
    with self.assertRaisesRegex(
        RuntimeError,
        r"(CUDA driver library is not available|CUDA_ERROR_INVALID_CONTEXT)",
    ):
      _profiler.start_capture()
    with self.assertRaisesRegex(
        RuntimeError,
        r"(CUDA driver library is not available|CUDA_ERROR_INVALID_CONTEXT)",
    ):
      _profiler.stop_capture()


if __name__ == "__main__":
  absltest.main()
