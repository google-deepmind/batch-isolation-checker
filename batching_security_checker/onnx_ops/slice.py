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

"""Define ONNX Slice operator."""

import functools

import jax
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import slice as slice_op

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=("starts", "ends", "axes", "steps"))
def onnx_taint_slice(*combined_args, starts, ends, axes, steps):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Slice."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="slice")

  if len(input_args) == 1 and len(taint_input_args) == 1:
    # in v1 of operator, the parameters were only passed as as attributes.
    pass
  else:
    if len(input_args) == 3:
      _, taint_starts, taint_ends = taint_input_args
      taint_axes = None
      taint_steps = None
    elif len(input_args) == 4:
      _, taint_starts, taint_ends, taint_axes = taint_input_args
      taint_steps = None
    elif len(input_args) == 5 and len(taint_input_args) == 5:
      _, taint_starts, taint_ends, taint_axes, taint_steps = taint_input_args
    else:
      raise ValueError(
          "Unexpected number of arguments:"
          f" {len(input_args)=} {len(taint_input_args)=}"
      )

    checkify.check(
        taint_propagation.is_all_untainted_deterministic(taint_starts),
        "slice: tainted or random starts not supported",
    )

    checkify.check(
        taint_propagation.is_all_untainted_deterministic(taint_ends),
        "slice: tainted or random ends not supported",
    )

    if taint_axes is not None:
      checkify.check(
          taint_propagation.is_all_untainted_deterministic(taint_axes),
          "slice: tainted or random axes not supported",
      )

    if taint_steps is not None:
      checkify.check(
          taint_propagation.is_all_untainted_deterministic(taint_steps),
          "slice: tainted or random steps not supported",
      )
  # We call original slice function but ensure that the arguments are untainted.
  result = slice_op.onnx_slice(
      *taint_input_args, starts=starts, ends=ends, axes=axes, steps=steps
  )
  return result


class TaintSlice(slice_op.Slice, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {slice_op.onnx_slice: onnx_taint_slice}

  @classmethod
  def data_handler(cls):
    return slice_op.Slice
