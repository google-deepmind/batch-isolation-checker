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

"""Define ONNX LayerNormalization operator."""

from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("LayerNormalization")
class LayerNormalization(handler.Handler):
  """Implementation of the ONNX LayerNormalization operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["axis"] = node.attrs.get("axis", -1)
    node.attrs_dict["epsilon"] = node.attrs.get("epsilon", 1e-05)
    node.attrs_dict["stash_type"] = node.attrs.get("stash_type", 1)

    node.attrs_dict["simplified"] = node.attrs.get("simplified", 0)

  @classmethod
  def version_17(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_17 LayerNormalization op."""
    cls._prepare(node, inputs, onnx_layernormalization)
    return onnx_layernormalization

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 LayerNormalization op."""
    cls._prepare(node, inputs, onnx_layernormalization)
    return onnx_layernormalization


@functools.partial(
    jax.jit, static_argnames=("axis", "epsilon", "stash_type", "simplified")
)
def onnx_layernormalization(*input_args, axis, epsilon, stash_type, simplified):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#LayerNormalization for more details."""

  # Follows the implementation in the cpu provider:
  # https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc

  # TODO: This implementation does not pass all the tests

  if len(input_args) == 2:
    data, scale = input_args
    bias = None
  elif len(input_args) == 3:
    data, scale, bias = input_args
    if simplified:
      raise ValueError("Simplified mode not supported with bias")
  else:
    raise ValueError(f"Unsupported number of inputs: {len(input_args)}")

  if stash_type == 0:
    original_dtype = jnp.dtype(data)
  elif stash_type == 1:
    original_dtype = jnp.dtype(data)
    data = jnp.astype(data, jnp.float32)
    scale = jnp.astype(scale, jnp.float32)
  else:
    raise ValueError(f"Unsupported stash_type: {stash_type}")

  # stage 1: normalize to zero mean and unit variance
  mean = jnp.mean(data, axis=axis, keepdims=True)

  # TODO: not 100% sure this is right
  mean_square = jnp.mean(data * data, axis=axis, keepdims=True)

  if simplified:
    mean_square = jnp.sqrt(mean_square + epsilon)
    normalized = data * jnp.reciprocal(mean_square)

    if bias is not None:
      raise ValueError("Bias not supported in simplified mode")

  else:
    mean_square = jnp.sqrt(mean_square + epsilon - mean * mean)
    normalized = (data - mean) * jnp.reciprocal(mean_square)

  normalized = jnp.astype(normalized, original_dtype)

  # stage 2: scale + [bias] output
  result = scale * normalized

  if bias is not None:
    result = result + bias

  return result
