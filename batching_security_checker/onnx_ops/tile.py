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

"""Define ONNX Tile operator."""

import functools

import jax
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import tile

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames="repeats")
def onnx_taint_tile(*combined_args, repeats):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Tile."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="tile")

  assert len(input_args) == 2, f"only version with two args is currently supported: {len(input_args)=}"
  assert len(taint_input_args) == 2

  _, taint_repeats = taint_input_args

  checkify.check(
      taint_propagation.is_all_untainted_deterministic(taint_repeats),
      "tile: tainted repeats not supported",
  )

  # We call original tile function but ensure that repeats is untainted.
  result = tile.onnx_tile(
      *taint_input_args, repeats=repeats
  )
  return result


class TaintTile(tile.Tile, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {tile.onnx_tile: onnx_taint_tile}

  @classmethod
  def data_handler(cls):
    return tile.Tile
