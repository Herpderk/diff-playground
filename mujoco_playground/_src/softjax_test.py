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

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest

import mujoco_playground
from mujoco_playground._src import softjax as sj


class SoftjaxTest(absltest.TestCase):

  def test_softness(self):
    self.assertEqual(sj.SOFTNESS, 0.01)

  def test_set_global_softness_updates_package_and_wrappers(self):
    try:
      mujoco_playground.set_global_softness(0.1)
      self.assertIs(type(sj.SOFTNESS), float)
      self.assertEqual(sj.SOFTNESS, 0.1)
      self.assertIn("set_global_softness", mujoco_playground.__all__)
      np.testing.assert_allclose(
          sj.relu(0.0), 0.1 * np.log(2.0), rtol=1e-5
      )
    finally:
      mujoco_playground.set_global_softness(0.01)

  def test_set_global_softness_rejects_invalid_values_without_changing_state(
      self
  ):
    try:
      mujoco_playground.set_global_softness(0.25)
      for softness in (0.0, -1.0, np.nan, np.inf, -np.inf):
        with self.subTest(softness=softness):
          with self.assertRaises(ValueError):
            mujoco_playground.set_global_softness(softness)
          self.assertEqual(sj.SOFTNESS, 0.25)
    finally:
      mujoco_playground.set_global_softness(0.01)

  def test_clip_at_lower_bound(self):
    np.testing.assert_allclose(
        sj.clip(jp.array(0.0), 0.0, 10000.0), 0.00693147, rtol=1e-5
    )
    np.testing.assert_allclose(
        sj.clip(jp.array(0.0), 0.0, 10000.0, softness=0.1),
        0.00693147,
        rtol=1e-5,
    )

  def test_abs(self):
    np.testing.assert_allclose(sj.abs(0.01), 0.00462117, rtol=1e-5)

  def test_comparisons(self):
    expected = 0.731059
    np.testing.assert_allclose(sj.greater(0.01, 0.0), expected, rtol=1e-5)
    np.testing.assert_allclose(
        sj.greater_equal(0.01, 0.0), expected, rtol=1e-5
    )
    np.testing.assert_allclose(sj.less(0.0, 0.01), expected, rtol=1e-5)
    np.testing.assert_allclose(
        sj.less_equal(0.0, 0.01), expected, rtol=1e-5
    )

  def test_st_comparisons_have_hard_forward_and_soft_gradients(self):
    comparisons = (
        (sj.greater_st, (0.0, 0.0, 1.0)),
        (sj.greater_equal_st, (0.0, 1.0, 1.0)),
        (sj.less_st, (1.0, 0.0, 0.0)),
        (sj.less_equal_st, (1.0, 1.0, 0.0)),
    )
    for comparison, expected in comparisons:
      with self.subTest(comparison=comparison.__name__):
        values = comparison(jp.array([-1.0, 0.0, 1.0]), 0.0)
        np.testing.assert_allclose(values, expected)
        _, gradient = jax.jvp(
            lambda x: comparison(x, 0.0),
            (jp.array(0.0),),
            (jp.array(1.0),),
        )
        self.assertNotEqual(float(gradient), 0.0)

  def test_st_comparisons_override_explicit_softness(self):
    comparisons = (
        sj.greater_st,
        sj.greater_equal_st,
        sj.less_st,
        sj.less_equal_st,
    )
    for comparison in comparisons:
      with self.subTest(comparison=comparison.__name__):
        gradients = []
        for softness in (0.1, 1.0):
          _, gradient = jax.jvp(
              lambda x: comparison(x, 0.0, softness=softness),
              (jp.array(0.0),),
              (jp.array(1.0),),
          )
          gradients.append(float(gradient))
        np.testing.assert_allclose(gradients[0], gradients[1])

        positional_gradient = jax.jvp(
            lambda x: comparison(x, 0.0, softness),
            (jp.array(0.0),),
            (jp.array(1.0),),
        )[1]
        np.testing.assert_allclose(positional_gradient, gradients[0])

  def test_reductions(self):
    values = jp.array([0.0, 0.01])
    np.testing.assert_allclose(sj.max(values), 0.01, rtol=1e-5)
    np.testing.assert_allclose(sj.min(values), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        sj.any(sj.greater(jp.zeros(4), 0.0)), 0.9375, rtol=1e-5
    )
    np.testing.assert_allclose(
        sj.any(sj.greater_st(jp.zeros(4), 0.0)), 0.0, atol=1e-5
    )

  def test_relu(self):
    np.testing.assert_allclose(sj.relu(0.0), 0.00693147, rtol=1e-5)

  def test_modes_are_forwarded(self):
    self.assertEqual(sj.clip(0.0, 0.0, 10000.0, mode="hard"), 0.0)
    self.assertEqual(sj.relu(-1.0, mode="hard"), 0.0)


if __name__ == "__main__":
  absltest.main()
