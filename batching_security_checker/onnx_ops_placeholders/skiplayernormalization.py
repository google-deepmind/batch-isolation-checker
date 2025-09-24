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

"""Define ONNX SkipLayerNormalization operator."""

from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node

# pylint: disable=unused-argument


@handler.register_op("SkipLayerNormalization", "com.microsoft")
class SkipLayerNormalization(handler.Handler):
  """Implementation of the ONNX Skip LayerNormalization operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["epsilon"] = node.attrs.get("epsilon", 1e-05)
    node.attrs_dict["simplified"] = node.attrs.get("simplified", 0)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 SkipLayerNormalization op."""
    cls._prepare(node, inputs, onnx_skiplayernormalization)
    return onnx_skiplayernormalization


@functools.partial(jax.jit, static_argnames=("epsilon", "simplified"))
def onnx_skiplayernormalization(*input_args, epsilon, simplified):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#LayerNormalization for more details."""

  # Follows the implementation in the cpu provider:
  # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/skip_layer_norm.cc

  # inputs:     input, skip, gamma, [beta] [bias]
  if len(input_args) == 3:
    data, skip, gamma = input_args
    beta = None
    bias = None
  elif len(input_args) == 4 and simplified:
    data, skip, gamma, bias = input_args
    beta = None
  elif len(input_args) == 5:
    data, skip, gamma, beta, bias = input_args
  else:
    raise ValueError(f"Unsupported number of inputs: {len(input_args)}")

  assert data.ndim == 3, "Input must be 3D: (batch, seqlen, hidden_size)"

  val = data + skip

  if bias is not None:
    val += bias

  # stage 1: normalize to zero mean and unit variance

  mean = jnp.mean(val, axis=-1, keepdims=True)

  # TODO: not 100% sure this is right
  mean_square = jnp.mean(val * val, axis=-1, keepdims=True)

  if simplified:
    mean_square = jnp.sqrt(mean_square + epsilon)
    normalized = val * jnp.reciprocal(mean_square)
    if beta:
      raise ValueError("Beta not supported in simplified mode")
  else:
    mean_square = jnp.sqrt(mean_square + epsilon - mean * mean)
    normalized = (val - mean) * jnp.reciprocal(mean_square)

  # stage 2: scale + [bias] output
  output = normalized * gamma

  if beta is not None:
    output += beta

  mean = None
  inv_std_var = None
  input_skip_bias_sum = val
  return output, mean, inv_std_var, input_skip_bias_sum
