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

"""Define ONNX ReduceSum operator."""

import functools

import jax
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import reducesum

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(
    jax.jit, static_argnames=('axes', 'keepdims', 'noop_with_empty_axes')
)
def onnx_taint_reducesum(
    *combined_args,
    axes=None,
    keepdims=1,
    noop_with_empty_axes=0,
):
  """The internal jax impl for onnx ReduceSum op."""
  _, taint_input_args = taint_handler.split_args(combined_args, op="reducesum")

  if len(taint_input_args) == 1:
    data = taint_input_args[0]
  elif len(taint_input_args) == 2:
    data, axes_taint = taint_input_args
    if axes_taint is not None:
      checkify.check(
          taint_propagation.is_all_untainted_deterministic(axes_taint),
          'axes must be untainted and deterministic',
      )
  else:
    raise ValueError(f'Unexpected number of arguments: {len(taint_input_args)}')
  if axes is None and noop_with_empty_axes > 0:
    return data
  out = taint_propagation.taint(data, axis=axes, keepdims=keepdims > 0)
  return out


class TaintReduceSum(reducesum.ReduceSum, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {reducesum.onnx_reducesum: onnx_taint_reducesum}

  @classmethod
  def data_handler(cls):
    return reducesum.ReduceSum
