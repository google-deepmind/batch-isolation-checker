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

"""Define ONNX GroupQueryAttention operator."""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import checkify
from jaxonnxruntime.core import config_class

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation
from batching_security_checker.onnx_ops_placeholders import groupqueryattention


config = config_class.config


@functools.partial(
    jax.jit,
    static_argnames=(
        "do_rotary",
        "kv_num_heads",
        "local_window_size",
        "num_heads",
        "rotary_interleaved",
        "scale",
        "smooth_softmax",
        "softcap",
        "total_sequence_length",
    ),
)
def onnx_taint_groupqueryattention(
    *combined_args,
    do_rotary,  # pylint: disable=unused-argument
    kv_num_heads,
    local_window_size,  # pylint: disable=unused-argument
    num_heads,
    rotary_interleaved,  # pylint: disable=unused-argument
    scale,  # pylint: disable=unused-argument
    smooth_softmax,  # pylint: disable=unused-argument
    softcap,  # pylint: disable=unused-argument
    total_sequence_length,
):
  """Taint propagation for GroupQueryAttention."""

  _, taint_input_args = taint_handler.split_args(combined_args, op="groupqueryattention")

  if len(taint_input_args) == 9:
    (
        query,
        key,
        value,
        past_key,
        past_value,
        seqlens_k,
        total_sequence_length_taint,
        _,
        _,
    ) = taint_input_args

    output_shape = get_output_shape(
        query=query,
        key=key,
        value=value,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
    )
  else:
    raise ValueError(f"Unsupported number of inputs: {len(taint_input_args)}")

  batch_size = jnp.shape(query)[0]
  assert output_shape[0] == batch_size

  taint = taint_propagation.binary_elementwise_taint(
      seqlens_k, total_sequence_length
  )
  taint = jnp.reshape(taint, (batch_size,))
  # taint: jax.Array = jnp.identity(taint)
  for tensor in [query, key, value, past_key, past_value]:
    if tensor and tensor is not None:
      dimensions = list(range(1, tensor.ndim))  # except for the batch dimension
      taint_tensor = taint_propagation.taint_reduction(tensor, dimensions)
      assert taint_tensor.ndim == 1
      assert jnp.shape(taint_tensor)[0] == batch_size

      assert jnp.shape(taint) == jnp.shape(
          taint_tensor
      ), f"{jnp.shape(taint)=} != {jnp.shape(taint_tensor)=}"
      taint = taint_propagation.binary_elementwise_taint(taint, taint_tensor)

  assert taint is not None
  assert jnp.shape(taint) == (batch_size,)

  output = jnp.reshape(taint, (batch_size, 1, 1))
  output = jnp.broadcast_to(output, output_shape)

  checkify.check(
      taint_propagation.is_all_untainted_deterministic(
          total_sequence_length_taint
      ),
      "total_sequence_length must be untainted, because the tensor shape"
      " (present_shape) is dependent on it",
  )

  present_shape = get_present_shape(
      output_shape=output_shape,
      past_key=past_key,
      past_value=past_value,
      total_sequence_length=total_sequence_length,
      num_heads=num_heads,
  )
  present_key = jnp.reshape(taint, (batch_size, 1, 1, 1))
  present_value = jnp.reshape(taint, (batch_size, 1, 1, 1))

  present_key = jnp.broadcast_to(present_key, present_shape)
  present_value = jnp.broadcast_to(present_value, present_shape)

  return output, present_key, present_value


def get_output_shape(query, key, value, num_heads, kv_num_heads):
  """Computes the shape of outputs."""

  assert query.ndim == 3, "Query must be 3D: (batch, seqlen, ?)"

  if key and value:
    batch_size, sequence_length, hidden_size = jnp.shape(query)
  elif not key and not value:
    # query is packed QKV
    batch_size, sequence_length, d = jnp.shape(query)
    head_size = d // (num_heads + 2 * kv_num_heads)
    assert d % (num_heads + 2 * kv_num_heads) == 0
    hidden_size = num_heads * head_size
  else:
    raise ValueError("Either both key and value must be provided or neither")

  return batch_size, sequence_length, hidden_size


def get_present_shape(
    output_shape, past_key, past_value, total_sequence_length, num_heads
):
  """Computes the shape of outputs: present_key, present_value."""

  batch_size, _, hidden_size = output_shape
  # head_size = hidden_size // num_heads
  assert hidden_size % num_heads == 0

  if (
      past_key
      and past_key is not None
      and past_value
      and past_value is not None
  ):

    assert (
        past_key.ndim == 4
    ), "Past key must be 4D: (batch, num_heads, sequence_length, head_size)"
    past_key_sequence_length = jnp.shape(past_key)[2]

    assert (
        past_value.ndim == 4
    ), "Past value must be 4D: (batch, num_heads, sequence_length, head_size)"
    past_value_sequence_length = jnp.shape(past_value)[2]

    assert (
        past_key_sequence_length == past_value_sequence_length
    ), "Past key and past value sequence lengths must be equal"

    # NOTE: total_sequence_length must be untainted for this to work.
    present_key_sequence_length = max(
        past_key_sequence_length, total_sequence_length
    )

    assert past_key.shape[1] == past_value.shape[1], (
        "Past key and past value must have the same number of heads, but got"
        f" {past_key.shape[1]=} != {past_value.shape[1]=}"
    )
    assert past_key.shape[3] == past_value.shape[3], (
        "Past key and past value must have the same head size, but got"
        f" {past_key.shape[3]=} != {past_value.shape[3]=}"
    )
    present_shape = (
        batch_size,
        past_key.shape[1],
        present_key_sequence_length,
        past_key.shape[3],
    )
  else:
    raise ValueError(
        "Either both past_key and past_value must be provided or neither"
    )

  return present_shape


class TaintGroupQueryAttention(
    groupqueryattention.GroupQueryAttention, taint_handler.TaintHandler
):
  """GroupQueryAttention operator taint handler."""

  @classmethod
  def jit_lookup(cls):
    return {
        groupqueryattention.onnx_groupqueryattention: (
            onnx_taint_groupqueryattention
        )
    }

  @classmethod
  def data_handler(cls):
    return groupqueryattention.GroupQueryAttention
