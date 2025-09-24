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

"""Define ONNX MatMulInteger operator."""

import functools

import jax
from batching_security_checker.onnx_ops_placeholders import matmulinteger

from batching_security_checker.onnx_ops import matmul
from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=())
def onnx_taint_matmulinteger(
    *combined_args,
):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#MatMulInteger."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="matmulinteger")

  if len(taint_input_args) == 2:
    a, b = taint_input_args
  elif len(taint_input_args) == 4:
    a, b, a_zero_point, b_zero_point = taint_input_args
    a = taint_propagation.binary_elementwise_taint(a, a_zero_point)
    b = taint_propagation.binary_elementwise_taint(b, b_zero_point)
  else:
    raise ValueError("????")

  out = matmul.onnx_taint_matmul(input_args[0], input_args[1], a, b)
  return out


class TaintMatMulInteger(matmulinteger.MatMulInteger, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {matmulinteger.onnx_matmulinteger: onnx_taint_matmulinteger}

  @classmethod
  def data_handler(cls):
    return matmulinteger.MatMulInteger
