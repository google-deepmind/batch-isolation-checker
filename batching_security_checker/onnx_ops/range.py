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

"""Define ONNX Range operator."""

import functools

import jax
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import range as range_op

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(
    jax.jit, static_argnames=("start", "limit", "delta", "dtype")
)
def onnx_taint_range(*combined_args, start, limit, delta, dtype):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Range for more details."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="range")

  assert len(input_args) == 3
  assert len(taint_input_args) == 3

  taint_start, taint_limit, taint_delta = taint_input_args

  # ensure that tensor shape is independent of tainted values
  checkify.check(
      taint_propagation.is_all_untainted_deterministic(taint_start),
      "range: tainted start not supported",
  )

  checkify.check(
      taint_propagation.is_all_untainted_deterministic(taint_limit),
      "range: tainted limit not supported",
  )

  checkify.check(
      taint_propagation.is_all_untainted_deterministic(taint_delta),
      "range: tainted delta not supported",
  )

  result = range_op.onnx_range(
      *input_args, start=start, limit=limit, delta=delta, dtype=dtype
  )

  return taint_propagation.identity_like(result)


class TaintRange(range_op.Range, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {range_op.onnx_range: onnx_taint_range}

  @classmethod
  def data_handler(cls):
    return range_op.Range
