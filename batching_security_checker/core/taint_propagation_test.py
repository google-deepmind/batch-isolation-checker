#
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Tests for core taint propagation logic."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax import typing as jtyping
from jax.experimental import checkify
import numpy as np

from batching_security_checker.core import config_class as taint_config
from batching_security_checker.core import taint_propagation

COLOR_LIMIT = (1 << 31) - 1  # 011..11


class TaintPropagationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # the tests assume uint64 taint
    taint_config.config.update("taint_dtype", "uint64")

  def assertEqualArray(self, a: jtyping.ArrayLike, b: jtyping.ArrayLike):
    self.assertEqual(jnp.shape(a), jnp.shape(b))
    self.assertEqual(jnp.dtype(a), jnp.dtype(b))
    self.assertTrue(jnp.all(a == b))

  def test_identity_element(self):

    id_element = taint_propagation.identity_element()
    self.assertEqual(id_element.dtype, jnp.uint64)
    id_element = np.asarray(id_element).tolist()

    # self.assertLen(id_element, 1)
    # id_element = id_element[0]

    id_element_np = taint_propagation.identity_element_concrete()
    id_element_np = np.asarray(id_element_np).tolist()

    # self.assertLen(id_element_np, 1)
    # id_element_np = id_element_np[0]

    self.assertEqual(id_element, id_element_np)
    self.assertEqual(id_element, (2**31) - 1)  # 32bit:011..1

  def test_not_a_color(self):
    value = taint_propagation.not_a_color()
    self.assertEqual(value.dtype, jnp.uint32)
    self.assertEqual(value, jnp.uint32((2**31) - 1))  # 32bit:011..1

  def test_from_color_to_taint(self):

    def _check(color, taint_ref):
      if not isinstance(color, list):
        color = [color]

      color_tensor = jnp.array(color, dtype=jnp.uint32)
      taint_tensor = taint_propagation.from_color_to_taint(color_tensor)
      self.assertEqual(taint_tensor.dtype, jnp.uint64)
      self.assertEqual(taint_tensor.shape, color_tensor.shape)

      if not isinstance(taint_ref, list):
        taint_ref = [taint_ref]

      taint_ref = jnp.array(taint_ref, dtype=jnp.uint64)
      self.assertEqual(taint_tensor.dtype, taint_ref.dtype)
      self.assertEqual(taint_tensor.shape, taint_ref.shape)
      self.assertTrue(jnp.all(taint_tensor == taint_ref))

    _check(color=0, taint_ref=0)
    self.assertEqual((1 << 31) + 1, 2147483648 + 1)
    _check(color=1, taint_ref=(1 << 31) + 1)
    _check(color=7, taint_ref=(7 << 31) + 7)
    _check(color=23548, taint_ref=(23548 << 31) + 23548)

    with self.assertRaises(checkify.JaxRuntimeError):
      _check(color=COLOR_LIMIT, taint_ref=0)

    with self.assertRaises(checkify.JaxRuntimeError):
      _check(color=COLOR_LIMIT + 10, taint_ref=0)

    color_max = COLOR_LIMIT - 1
    _check(color=color_max, taint_ref=(color_max << 31) + color_max)

    _check(color=[0, 2, 3], taint_ref=[0, (2 << 31) + 2, (3 << 31) + 3])

    _check(
        color=[[10, 11], [20, 21], [30, 31]],
        taint_ref=[
            [(10 << 31) + 10, (11 << 31) + 11],
            [(20 << 31) + 20, (21 << 31) + 21],
            [(30 << 31) + 30, (31 << 31) + 31],
        ],
    )

    with self.assertRaises(checkify.JaxRuntimeError):
      _check(color=[COLOR_LIMIT, 3], taint_ref=[0, 0])

  def test_from_color_taint_round_trip(self):
    def _check(color):
      if not isinstance(color, list):
        color = [color]
      color = jnp.array(color, dtype=jnp.uint32)
      taint = taint_propagation.from_color_to_taint(color)
      color_actual = taint_propagation.from_taint_to_color(taint)

      self.assertEqual(color_actual.dtype, color.dtype)
      self.assertEqual(color_actual.shape, color.shape)
      self.assertTrue(jnp.all(color_actual == color))

    _check(color=0)
    _check(color=1)
    _check(color=7)
    _check(color=23548)
    _check(color=[0, 2, 3])
    _check(color=[[10, 11], [20, 21], [30, 31]])

    with self.assertRaises(checkify.JaxRuntimeError):
      _check(color=COLOR_LIMIT)

    _check(color=COLOR_LIMIT - 1)

  def test_from_taint_to_unique_interfering_colors(self):

    def _extract_and_check(taint: jax.Array, colors1: list[int], colors2=None):
      info = taint_propagation.from_taint_to_unique_interfering_colors(taint)
      info = sorted(info, key=lambda x: (x["min"], x["max"]))

      if colors2 is None:
        colors2 = colors1

      ref = {
          (min(c1, c2), max(c1, c2))
          for c1, c2 in zip(colors1, colors2, strict=True)
          if c1 != c2
      }

      ref = sorted(
          [{"min": x[0], "max": x[1]} for x in ref],
          key=lambda x: (x["min"], x["max"]),
      )

      self.assertEqual(info, ref)

    colors1 = [0, 1, 2, 2, 23456, 23456]
    colors1_arr = jnp.array(colors1, dtype=jnp.uint32)
    taint1_arr = taint_propagation.from_color_to_taint(colors1_arr)

    # Without interference
    _extract_and_check(taint1_arr, colors1)

    # With interference
    colors2 = [0, 23456, 2, 3, 23456, 1]
    colors2_arr = jnp.array(colors2, dtype=jnp.uint32)
    taint2_arr = taint_propagation.from_color_to_taint(colors2_arr)

    taint_combined = taint_propagation.binary_elementwise_taint(
        taint1_arr, taint2_arr
    )
    _extract_and_check(taint_combined, colors1, colors2)

  def test_is_untained(self):

    x = jnp.full(
        (1, 2, 3), taint_propagation.identity_element(), dtype=jnp.uint64
    )
    self.assertEqual(taint_propagation.is_all_untainted(x), jnp.array(True))

    x = x.at[(0, 0, 1)].set(
        taint_propagation.from_color_to_taint(jnp.array(1, dtype=jnp.uint32))
    )

    self.assertEqual(taint_propagation.is_all_untainted(x), jnp.array(False))

  def test_identity_taint(self):
    def _check(colors1: list[int], colors2: list[int]):

      colors1_arr = jnp.array(colors1, dtype=jnp.uint32)
      taint1_arr = taint_propagation.from_color_to_taint(colors1_arr)

      colors2_arr = jnp.array(colors2, dtype=jnp.uint32)
      taint2_arr = taint_propagation.from_color_to_taint(colors2_arr)

      combined = taint_propagation.binary_elementwise_taint(
          taint1_arr, taint2_arr
      )

      out = taint_propagation.identity_taint(combined)

      self.assertEqual(jnp.shape(out), jnp.shape(combined))
      self.assertEqual(jnp.dtype(out), jnp.dtype(combined))
      self.assertTrue(jnp.all(out == combined))

    colors1 = [0, 2, 3]
    colors2 = [0, 1, 2]
    _check(colors1, colors1)
    _check(colors2, colors2)
    _check(colors1, colors2)

  def test_set_random(self):

    colors = [0, 0, 1, 1, 4, 5, 6, 7, 8, 9]
    colors = jnp.array(colors, dtype=jnp.uint32)
    x_tainted = taint_propagation.from_color_to_taint(colors)

    before = taint_propagation.is_random(x_tainted)

    self.assertTrue(
        jnp.all(before == jnp.full_like(colors, False, dtype=jnp.bool))
    )
    out = taint_propagation.set_random(x_tainted)
    self.assertEqual(jnp.dtype(out), jnp.dtype(x_tainted))
    self.assertEqual(jnp.shape(out), jnp.shape(x_tainted))

    after = taint_propagation.is_random(out)
    self.assertEqual(jnp.dtype(after), jnp.bool)
    self.assertEqual(jnp.shape(after), jnp.shape(x_tainted))
    self.assertTrue(
        jnp.all(after == jnp.full_like(colors, True, dtype=jnp.bool))
    )

  def test_binary_elementwise_taint(self):
    def encode(colors):
      colors_arr = jnp.array(colors, dtype=jnp.uint32)
      return taint_propagation.from_color_to_taint(colors_arr)

    colors1 = [0, 1, 3]
    colors2 = [0, 1, 2]
    colors3 = 1

    taint1 = encode(colors1)
    taint2 = encode(colors2)
    taint3 = encode(colors3)

    self.assertEqualArray(
        taint1, taint_propagation.binary_elementwise_taint(taint1, taint1)
    )
    self.assertEqualArray(
        taint2, taint_propagation.binary_elementwise_taint(taint2, taint2)
    )
    self.assertEqualArray(
        taint3, taint_propagation.binary_elementwise_taint(taint3, taint3)
    )

    mix1 = taint_propagation.binary_elementwise_taint(taint1, taint2)
    self.assertEqual(mix1[0], taint1[0])
    self.assertEqual(mix1[1], taint1[1])
    self.assertEqual(mix1[2], jnp.array((3 << 31) + 2))

    mix2 = taint_propagation.binary_elementwise_taint(taint1, taint3)
    self.assertEqual(mix2[0], jnp.array((1 << 31) + 0))
    self.assertEqual(mix2[1], taint1[1])
    self.assertEqual(mix2[2], jnp.array((3 << 31) + 1))

    # check config flags propagation
    taint1_random = taint_propagation.set_random(taint1)
    taint2_random = taint_propagation.set_random(taint2)
    mix = taint_propagation.binary_elementwise_taint(taint1_random, taint2)
    self.assertFalse(jnp.any(taint_propagation.is_random(taint2)))
    self.assertTrue(jnp.all(taint_propagation.is_random(mix)))

    mix = taint_propagation.binary_elementwise_taint(taint1, taint2_random)
    self.assertFalse(jnp.any(taint_propagation.is_random(taint1)))
    self.assertTrue(jnp.all(taint_propagation.is_random(mix)))

    mix = taint_propagation.binary_elementwise_taint(
        taint1_random, taint2_random
    )
    self.assertTrue(jnp.all(taint_propagation.is_random(mix)))

    mix = taint_propagation.binary_elementwise_taint(
        taint1_random, taint1_random
    )
    self.assertTrue(jnp.all(taint_propagation.is_random(mix)))

  def test_has_single_taint(self):

    def encode(colors):
      colors_arr = jnp.array(colors, dtype=jnp.uint32)
      return taint_propagation.from_color_to_taint(colors_arr)

    colors1 = [0, 1, 3]
    colors2 = 3

    taint1 = encode(colors1)
    taint2 = encode(colors2)

    self.assertEqual(
        jnp.all(taint_propagation.has_all_single_taint(taint1)), jnp.array(True)
    )

    combined = taint_propagation.binary_elementwise_taint(taint1, taint2)
    out = taint_propagation.has_all_single_taint(combined)
    self.assertEqualArray(out, jnp.array(False))


if __name__ == "__main__":
  absltest.main()
