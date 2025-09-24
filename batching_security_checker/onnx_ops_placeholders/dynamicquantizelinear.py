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
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils

# TODO [nku] test the operator!!!!!!!!
@handler.register_op("DynamicQuantizeLinear")
class DynamicQuantizeLinear(handler.Handler):
  """Implementation of the ONNX DynamicQuantizeLinear operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_node.update_node_attr_dict_with_jax_func_kwargs(node, onnx_jax_impl)
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_11(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_11 DynamicQuantizeLinear op."""
    cls._prepare(node, inputs, onnx_dynamicquantizelinear)
    return onnx_dynamicquantizelinear


@functools.partial(jax.jit, static_argnames=())
def onnx_dynamicquantizelinear(*input_args):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#DynamicQuantizeLinear for more details."""

  qmin = 0
  qmax = 255
  dtype = jnp.uint8

  if len(input_args) == 1:
    x = input_args[0]
  else:
    raise ValueError(f"Unsupported number of inputs: {len(input_args)}")

  max_x = jnp.max(x, axis=None)
  min_x = jnp.min(x, axis=None)
  y_scale = (jnp.maximum(0, max_x) - jnp.minimum(0, min_x)) / (qmax - qmin)
  # y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)

  intermediate_zero_point = qmin - (min_x / y_scale)
  y_zero_point = jnp.round(jnp.clip(intermediate_zero_point, qmin, qmax))
  y_zero_point = y_zero_point.astype(dtype)
  #y_zero_point = cast(round(saturate(itermediate_zero_point)))

  # quantize linear
  xi = jnp.round(x / y_scale)
  xi = jnp.rint(xi).astype(jnp.int32)
  xi = xi + y_zero_point
  y =  jnp.clip(xi, 0, 255).astype(dtype)

  return y, y_scale, y_zero_point
