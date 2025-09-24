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

"""Define ONNX RotaryEmbedding operator."""

from collections.abc import Callable, Sequence
import functools
from typing import Any
import warnings

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("RotaryEmbedding", "com.microsoft")
class RotaryEmbedding(handler.Handler):
  """Implementation of the ONNX RotaryEmbedding operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):

    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 RotaryEmbedding op."""
    cls._prepare(node, inputs, onnx_rotaryembedding)
    return onnx_rotaryembedding


@functools.partial(
    jax.jit,
    static_argnames=(
        "interleaved",
        "is_packed_batching",
        "num_heads",
        "rotary_embedding_dim",
        "scale",
    ),
)
def onnx_rotaryembedding(
    *input_args,
    interleaved,
    is_packed_batching,
    num_heads,
    rotary_embedding_dim,
    scale,
):
  """https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftrotaryembedding for more details."""

  warnings.warn(
      "Using placeholder implementation of RotaryEmbedding for inference"
      " that outputs nan tensors in the correct shape. This can be safely used"
      " for taint propagation but not for inference."
  )

  if len(input_args) == 4:
    (
        data,
        position_ids,
        cos_cache,
        sin_cache,
    ) = input_args
  else:
    raise ValueError(f"Unsupported number of inputs: {len(input_args)}")
  output = jnp.full_like(data, jnp.nan)
  return output
