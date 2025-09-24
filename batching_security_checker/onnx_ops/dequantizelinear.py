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

"""Define ONNX DequantizeLinear operator."""

import functools

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import dequantizelinear

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames="axis")
def onnx_taint_dequantizelinear(*combined_args, axis=None):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#DequantizeLinear."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="dequantizelinear")

  pad_value = taint_propagation.identity_like(input_args[1])

  x, x_scale, x_zero_point = onnx_node.pad_sequence(
      taint_input_args, 3, pad_value=pad_value
  )

  axis = 1 if axis is None else axis

  x_scale = dequantizelinear.reshape_input(x_scale, jnp.shape(x), axis)
  x_zero_point = dequantizelinear.reshape_input(x_zero_point, jnp.shape(x), axis)

  result = taint_propagation.binary_elementwise_taint(x, x_zero_point)
  result = taint_propagation.binary_elementwise_taint(result, x_scale)

  return result


class TaintDequantizeLinear(
    dequantizelinear.DequantizeLinear, taint_handler.TaintHandler
):

  @classmethod
  def jit_lookup(cls):
    return {dequantizelinear.onnx_dequantizelinear: onnx_taint_dequantizelinear}

  @classmethod
  def data_handler(cls):
    return dequantizelinear.DequantizeLinear
