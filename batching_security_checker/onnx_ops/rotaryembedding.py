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

import functools

import jax
from jax import numpy as jnp

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation
from batching_security_checker.onnx_ops_placeholders import rotaryembedding

# pylint: disable=unused-argument


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
def onnx_taint_rotaryembedding(
    *combined_args,
    interleaved,
    is_packed_batching,
    num_heads,
    rotary_embedding_dim,
    scale,
):
  """The jax impl for taint RotaryEmbedding op."""

  _, taint_input_args = taint_handler.split_args(combined_args, op="rotaryembedding")

  if len(taint_input_args) == 4:
    (
        data,
        position_ids,
        _,  # cos_cache
        _,  # sin_cache,
    ) = taint_input_args
  else:
    raise ValueError(f"Unsupported number of inputs: {len(taint_input_args)}")

  assert (
      data.ndim == 3 or data.ndim == 4
  ), "Data must be 3D or 4D: (batch_size, ...)"
  batch_size = jnp.shape(data)[0]
  dimensions = list(range(1, data.ndim))  # except for the batch dimension
  taint_data = taint_propagation.taint_reduction(data, dimensions)

  assert position_ids.ndim == 2, "Position ids must be 2D: (batch, seqlen)"
  taint_pos = taint_propagation.taint_reduction(position_ids, [1])

  assert jnp.shape(taint_data) == jnp.shape(
      taint_pos
  ), "Taint data and taint pos must have the same shape"
  taint = taint_propagation.binary_elementwise_taint(taint_data, taint_pos)

  if data.ndim == 3:
    output = jnp.reshape(taint, (batch_size, 1, 1))
  elif data.ndim == 4:
    output = jnp.reshape(taint, (batch_size, 1, 1, 1))
  else:
    raise ValueError(f"Unsupported data ndim: {data.ndim}")

  output_shape = jnp.shape(data)
  output = jnp.broadcast_to(output, output_shape)

  return output


class TaintRotaryEmbedding(
    rotaryembedding.RotaryEmbedding, taint_handler.TaintHandler
):

  @classmethod
  def jit_lookup(cls):
    return {rotaryembedding.onnx_rotaryembedding: onnx_taint_rotaryembedding}

  @classmethod
  def data_handler(cls):
    return rotaryembedding.RotaryEmbedding
