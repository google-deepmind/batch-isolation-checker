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

"""Define ONNX DynamicQuantizeLinear operator."""

import functools

import jax
from jax import numpy as jnp
from batching_security_checker.onnx_ops_placeholders import dynamicquantizelinear

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation

@functools.partial(jax.jit, static_argnames=())
def onnx_taint_dynamicquantizelinear(*combined_args):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#DynamicQuantizeLinear."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="dynamicquantizelinear")

  if len(taint_input_args) == 1 and len(input_args) == 1:
    x = taint_input_args[0]
  else:
    raise ValueError(f"Unknown number of inputs: {len(taint_input_args)}")

  reduce_scalar = taint_propagation.taint_reduction(x, tuple(range(0, jnp.ndim(x))))

  y_scale = reduce_scalar
  y_zero_point = reduce_scalar
  y = jnp.full_like(x, reduce_scalar)

  return y, y_scale, y_zero_point


class TaintDynamicQuantizeLinear(
    dynamicquantizelinear.DynamicQuantizeLinear, taint_handler.TaintHandler
):

  @classmethod
  def jit_lookup(cls):
    return {dynamicquantizelinear.onnx_dynamicquantizelinear: onnx_taint_dynamicquantizelinear}

  @classmethod
  def data_handler(cls):
    return dynamicquantizelinear.DynamicQuantizeLinear
