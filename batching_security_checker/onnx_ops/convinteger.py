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

"""Define ONNX ConvInteger operator."""

import functools
from typing import Any, Optional

import jax
from batching_security_checker.onnx_ops_placeholders import convinteger

from batching_security_checker.onnx_ops import conv
from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(
    jax.jit,
    static_argnames=("group", "kernel_shape", "pads", "strides", "dilations"),
)
def onnx_taint_convinteger(
    *combined_args,
    group: int = 1,
    kernel_shape: Optional[tuple[int, ...]] = None,
    pads: Any = "VALID",
    strides: Optional[tuple[int, ...]] = None,
    dilations: Optional[tuple[int, ...]] = None,
) -> jax.Array:
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ConvInteger."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="convinteger")

  if len(taint_input_args) == 2:
    x, w = taint_input_args
  elif len(taint_input_args) == 3:
    x, w, x_zero_point = taint_input_args
    x = taint_propagation.binary_elementwise_taint(x, x_zero_point)
  elif len(taint_input_args) == 4:
    x, w, x_zero_point, w_zero_point = taint_input_args
    x = taint_propagation.binary_elementwise_taint(x, x_zero_point)
    w = taint_propagation.binary_elementwise_taint(w, w_zero_point)
  else:
    raise ValueError("????")


  inputs = (input_args[0], input_args[1], x, w)
  out = conv.onnx_taint_conv(*inputs, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations)
  return out


class TaintConvInteger(convinteger.ConvInteger, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {convinteger.onnx_convinteger: onnx_taint_convinteger}

  @classmethod
  def data_handler(cls):
    return convinteger.ConvInteger
