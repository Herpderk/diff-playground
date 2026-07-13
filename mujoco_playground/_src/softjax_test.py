# Copyright 2025 DeepMind Technologies Limited
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
"""Tests for the project softjax adapter."""

from absl.testing import absltest
import jax.numpy as jp
import numpy as np

from mujoco_playground._src import softjax as sj


class SoftjaxTest(absltest.TestCase):

  def test_softness(self):
    self.assertEqual(sj.SOFTNESS, 5e-3)

  def test_clip_at_lower_bound(self):
    np.testing.assert_allclose(
        sj.clip(jp.array(0.0), 0.0, 10000.0), 0.00346574, rtol=1e-5
    )
    np.testing.assert_allclose(
        sj.clip(jp.array(0.0), 0.0, 10000.0, softness=0.1),
        0.00346574,
        rtol=1e-5,
    )

  def test_abs(self):
    np.testing.assert_allclose(sj.abs(0.01), 0.00761594, rtol=1e-5)

  def test_comparisons(self):
    expected = 0.880797
    np.testing.assert_allclose(sj.greater(0.01, 0.0), expected, rtol=1e-5)
    np.testing.assert_allclose(
        sj.greater_equal(0.01, 0.0), expected, rtol=1e-5
    )
    np.testing.assert_allclose(sj.less(0.0, 0.01), expected, rtol=1e-5)
    np.testing.assert_allclose(
        sj.less_equal(0.0, 0.01), expected, rtol=1e-5
    )

  def test_reductions(self):
    values = jp.array([0.0, 0.01])
    np.testing.assert_allclose(sj.max(values), 0.01, rtol=1e-5)
    np.testing.assert_allclose(sj.min(values), 0.0, rtol=1e-5)

  def test_relu(self):
    np.testing.assert_allclose(sj.relu(0.0), 0.00346574, rtol=1e-5)

  def test_modes_are_forwarded(self):
    self.assertEqual(sj.clip(0.0, 0.0, 10000.0, mode="hard"), 0.0)
    self.assertEqual(sj.relu(-1.0, mode="hard"), 0.0)


if __name__ == "__main__":
  absltest.main()
