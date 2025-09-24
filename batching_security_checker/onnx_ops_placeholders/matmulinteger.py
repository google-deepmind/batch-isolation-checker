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
from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils

# TODO: Needs to be tested
@handler.register_op("MatMulInteger")
class MatMulInteger(handler.Handler):
  """Implementation of the ONNX MatMulInteger operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_10 MatMulInteger op."""
    cls._prepare(node, inputs, onnx_matmulinteger)
    return onnx_matmulinteger

@functools.partial(jax.jit, static_argnames=())
def onnx_matmulinteger(*input_args):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#MatMulInteger."""
  if len(input_args) == 2:
    a, b = input_args
    a_zero_point = 0
    b_zero_point = 0
  elif len(input_args) == 4:
    a, b, a_zero_point, b_zero_point  = input_args
  else:
    raise ValueError(f"Unexpected number of args:  {len(input_args)=}")
  a = jnp.astype(a, jnp.int32) - jnp.astype(a_zero_point, jnp.int32)
  b = jnp.astype(b, jnp.int32) - jnp.astype(b_zero_point, jnp.int32)
  return jnp.matmul(a, b)
