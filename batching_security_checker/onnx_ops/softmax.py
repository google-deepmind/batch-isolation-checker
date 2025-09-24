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

"""Define ONNX Softmax operator."""

import functools

import jax
from jax import numpy as jnp
from jaxonnxruntime.onnx_ops import softmax

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=("axis",))
def onnx_taint_softmax(*combined_args, axis):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Softmax."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="softmax")

  assert len(input_args) == 1
  assert len(taint_input_args) == 1

  x = taint_input_args[0]

  axis = x.ndim - 1 if axis == -1 else axis
  taint = taint_propagation.taint_reduction(x, [axis])
  taint = jnp.expand_dims(taint, axis=axis)

  assert jnp.ndim(x) == jnp.ndim(taint), f"{jnp.shape(x)=}  {jnp.shape(taint)=}"

  # broadcast the reduced taint result to the original tensor shape
  out = jnp.broadcast_to(taint, jnp.shape(x))

  return out


class TaintSoftmax(softmax.Softmax, taint_handler.TaintHandler):
  """Taint handler for Softmax operator."""

  @classmethod
  def jit_lookup(cls):
    return {softmax.onnx_softmax: onnx_taint_softmax}

  @classmethod
  def data_handler(cls):
    return softmax.Softmax
