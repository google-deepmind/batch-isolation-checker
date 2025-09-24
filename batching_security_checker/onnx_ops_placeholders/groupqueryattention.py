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

from collections.abc import Callable, Sequence
import functools
from typing import Any
import warnings

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import config_class
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils

# pylint: disable=unused-argument


config = config_class.config


@handler.register_op("GroupQueryAttention", "com.microsoft")
class GroupQueryAttention(handler.Handler):
  """Implementation of the ONNX GroupQueryAttention operator.

   Abbreviation and Meanings: (src:
   https://github.com/microsoft/onnxruntime/blob/a4eb8f27b6e51dec41f943b614702dd114731e13/onnxruntime/contrib_ops/cpu/bert/attention_base.cc)
     B:    batch_size
     S:    sequence_length (input sequence length of query)
     P:    past_sequence_length (past sequence length of key or value)
     L:    kv_sequence_length (input sequence length of key or value)
     M:    max_sequence_length
     T:    total_sequence_length = past_sequence_length + kv_sequence_length
     N:    num_heads
     H:    head size for Q and K, aka q_head_size or k_head_size or qk_head_size
     H_v:  v_head_size
     D_i:  input hidden size
     D:    hidden size for Q and K (D = N * H), aka q_hidden_size or
     k_hidden_size or qk_hidden_size
     D_v:  v_hidden_size = num_heads * v_head_size

  When past state is used, Q, K and V should have same hidden size (unless we
  split it into past_key and past_value).
  """

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):

    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

    assert len(node.inputs) == 9
    # total_sequence_length = node.inputs[6]
    if config.jaxort_only_allow_initializers_as_static_args:

      if node.inputs[6] not in node.context_graph.get_constant_dict():
        raise ValueError(
            f"{node.inputs[6]} is not constant but used as a static argument "
            "`shape` when `jax.jit` the `GroupQueryAttention` operator. "
            "The jitted function gives wrong results if its value changes."
        )
      node.attrs_dict["total_sequence_length"] = (
          node.context_graph.get_constant_dict()[node.inputs[6]].tolist()
      )
    else:
      node.attrs_dict["total_sequence_length"] = inputs[6].tolist()

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 GroupQueryAttention op."""
    cls._prepare(node, inputs, onnx_groupqueryattention)
    return onnx_groupqueryattention


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
def onnx_groupqueryattention(
    *input_args,
    do_rotary,
    kv_num_heads,
    local_window_size,
    num_heads,
    rotary_interleaved,
    scale,
    smooth_softmax,
    softcap,
    total_sequence_length,
):
  """https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftgroupqueryattention for more details."""

  warnings.warn(
      "Using placeholder implementation of GroupQueryAttention for inference"
      " that outputs nan tensors in the correct shape. This can be safely used"
      " for taint propagation but not for inference."
  )

  if len(input_args) == 9:
    (
        query,
        key,
        value,
        past_key,
        past_value,
        _,  # seqlens_k,
        _,  # total_sequence_length
        _,  # cos_cache,
        _,  # sin_cache,
    ) = input_args

    output_shape = get_output_shape(
        query=query,
        key=key,
        value=value,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
    )
  else:
    raise ValueError(f"Unsupported number of inputs: {len(input_args)}")

  # max_sequence_length or past_sequence_length + kv_sequence_length

  #  (batch_size, sequence_length, hidden_size)   BSD  (D = N * H)
  output = jnp.full(output_shape, jnp.nan, dtype=jnp.dtype(query))

  present_shape = get_present_shape(
      output_shape=output_shape,
      past_key=past_key,
      past_value=past_value,
      total_sequence_length=total_sequence_length,
      num_heads=num_heads,
  )
  present_key = jnp.full(present_shape, jnp.nan, dtype=jnp.dtype(query))
  present_value = jnp.full(present_shape, jnp.nan, dtype=jnp.dtype(query))

  return output, present_key, present_value


def get_output_shape(query, key, value, num_heads, kv_num_heads):
  """Computes the shape of outputs."""

  assert query.ndim == 3, "Query must be 3D: (batch, seqlen, ?)"

  if key and value:
    batch_size, sequence_length, hidden_size = query.shape
  elif not key and not value:
    # query is packed QKV
    batch_size, sequence_length, d = query.shape
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
    past_key_sequence_length = past_key.shape[2]

    assert (
        past_value.ndim == 4
    ), "Past value must be 4D: (batch, num_heads, sequence_length, head_size)"
    past_value_sequence_length = past_value.shape[2]

    assert (
        past_key_sequence_length == past_value_sequence_length
    ), "Past key and past value sequence lengths must be equal"

    # TODO: total_sequence_length MUST BE UNTAINTED FOR THIS TO WORK
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
