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

"""Define ONNX Reshape operator."""

import functools

import jax
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import reshape

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=("shape", "allowzero"))
def onnx_taint_reshape(*combined_args, shape, allowzero):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Reshape."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="reshape")

  assert len(input_args) == 2
  assert len(taint_input_args) == 2

  _, taint_input_shape = taint_input_args

  checkify.check(
      taint_propagation.is_all_untainted_deterministic(taint_input_shape),
      "reshape: tainted input shape not supported",
  )

  # We call original reshape function but ensure that the shape is untainted.
  result = reshape.onnx_reshape(
      *taint_input_args, shape=shape, allowzero=allowzero
  )
  return result


class TaintReshape(reshape.Reshape, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {reshape.onnx_reshape: onnx_taint_reshape}

  @classmethod
  def data_handler(cls):
    return reshape.Reshape
