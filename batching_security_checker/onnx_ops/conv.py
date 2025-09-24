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

"""Define ONNX Conv operator."""

import functools
from typing import Any, Optional

import jax
from jax import numpy as jnp
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import conv

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(
    jax.jit,
    static_argnames=("group", "kernel_shape", "pads", "strides", "dilations"),
)
def onnx_taint_conv(
    *combined_inputs,
    group: int = 1,
    kernel_shape: Optional[tuple[int, ...]] = None,
    pads: Any = "VALID",
    strides: Optional[tuple[int, ...]] = None,
    dilations: Optional[tuple[int, ...]] = None,
) -> jax.Array:
  """The internal jax impl for onnx Conv op."""

  _, taint_inputs = taint_handler.split_args(combined_inputs, op="conv")

  assert (
      len(taint_inputs) == 2 or len(taint_inputs) == 3
  ), f"{len(taint_inputs)=} split not correct"

  # We are calling the original conv function with the tainted inputs.
  #   to get the shape of the output.
  #   -> Would be more efficient to use:
  # https://jax.readthedocs.io/en/latest/_autosummary/jax.eval_shape.html
  # NOTE: We always need to pass the additional arguments key=value syntax.
  conv_out = conv.onnx_conv(
      *taint_inputs,
      group=group,
      kernel_shape=kernel_shape,
      pads=pads,
      strides=strides,
      dilations=dilations,
  )

  assert len(taint_inputs) == 2 or len(taint_inputs) == 3
  if len(taint_inputs) == 2:
    x, w = taint_inputs
    # b = None
  else:
    x, w, b = taint_inputs
    checkify.check(
        taint_propagation.is_all_untainted(b),
        "conv: tainted bias not supported",
    )

  dimensions = [i for i in range(0, jnp.ndim(w))]
  kernel_taint = taint_propagation.taint_reduction(w, dimensions)
  kernel_taint = jnp.expand_dims(kernel_taint, axis=0)  # add a batch dimension

  # NOTE: We could make taint propagation more fine-grained with:
  # window_reduce (see maxpool)
  # kernel_shape = kernel_shape or w.shape
  # spatial_size = w.ndim - 2
  # strides = strides or tuple([1] * spatial_size)

  # reduce all dimensions except the batch dimension (0th index)
  dimensions = [i for i in range(1, jnp.ndim(x))]
  data_out = taint_propagation.taint_reduction(x, dimensions)

  out = taint_propagation.binary_elementwise_taint(data_out, kernel_taint)

  # because broadcast rules are from dim right to left
  shape = [1] * jnp.ndim(x)
  shape[0] = jnp.shape(x)[0]
  out = jnp.reshape(out, shape)
  out = jnp.broadcast_to(out, jnp.shape(conv_out))

  return out


class TaintConv(taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {conv.onnx_conv: onnx_taint_conv}

  @classmethod
  def data_handler(cls):
    return conv.Conv
