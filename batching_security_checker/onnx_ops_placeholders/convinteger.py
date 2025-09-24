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
from collections.abc import Callable, Sequence
import functools
from typing import Any, Optional

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils

from jaxonnxruntime.onnx_ops import conv

# TODO: Needs to be tested (not sure simply subtracting the zero point is sufficient)
@handler.register_op("ConvInteger")
class ConvInteger(handler.Handler):
  """Implementation of the ONNX ConvInteger operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    conv.Conv._prepare(node, inputs, onnx_jax_impl)

  @classmethod
  def version_10(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_10 ConvInteger op."""
    cls._prepare(node, inputs, onnx_convinteger)
    return onnx_convinteger

@functools.partial(
    jax.jit,
    static_argnames=("group", "kernel_shape", "pads", "strides", "dilations"),
)
def onnx_convinteger(
    *inputs,
    group: int = 1,
    kernel_shape: Optional[tuple[int, ...]] = None,
    pads: Any = "VALID",
    strides: Optional[tuple[int, ...]] = None,
    dilations: Optional[tuple[int, ...]] = None,
) -> jax.Array:
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ConvInteger."""
  if len(inputs) == 2:
    x, w = inputs
    x_zero_point = 0
    w_zero_point = 0
  elif len(inputs) == 3:
    x, w, x_zero_point  = inputs
    w_zero_point = 0
  elif len(inputs) == 4:
    x, w, x_zero_point, w_zero_point  = inputs
  else:
    raise ValueError(f"Unexpected number of arguments: {len(inputs)=}")



  x = jnp.astype(x, jnp.int32) - jnp.astype(x_zero_point, jnp.int32)
  w = jnp.astype(w, jnp.int32) - jnp.astype(w_zero_point, jnp.int32)
  out = conv.onnx_conv(x,w, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides, dilations=dilations)
  return out.astype(jnp.int32)
