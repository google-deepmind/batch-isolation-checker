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

"""Define ONNX Squeeze operator."""

import functools

import jax
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import squeeze

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames='axis')
def onnx_taint_squeeze(*combined_args, axis):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Squeeze."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="squeeze")

  if len(input_args) == 1:
    assert len(taint_input_args) == 1
  elif len(input_args) == 2:
    assert len(taint_input_args) == 2
    _, taint_axes = taint_input_args

    checkify.check(
        taint_propagation.is_all_untainted_deterministic(taint_axes),
      "squeeze: tainted axes not supported",
    )
  else:
    raise ValueError(f"Unexpected number of arguments: {len(input_args)=}")

  # We call original squeeze function but ensure that the axes is untainted.
  result = squeeze.onnx_squeeze(
      *taint_input_args, axis=axis
  )
  return result


class TaintSqueeze(squeeze.Squeeze, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {squeeze.onnx_squeeze: onnx_taint_squeeze}

  @classmethod
  def data_handler(cls):
    return squeeze.Squeeze
