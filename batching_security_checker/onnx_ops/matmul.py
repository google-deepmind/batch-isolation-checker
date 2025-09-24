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

"""Define ONNX MatMul operator."""

import functools

import jax
from jax import numpy as jnp
from jaxonnxruntime.onnx_ops import matmul

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=())
def onnx_taint_matmul(
    *combined_args,
):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#MatMul."""

  _, taint_input_args = taint_handler.split_args(combined_args, op="matmul")

  assert len(taint_input_args) == 2
  a, b = taint_input_args

  if a.ndim == 1 or b.ndim == 1:
    # Option 1: Dot Product (N,)  (N,) -> (1,)
    # TODO: Add missing features in matmul
    raise ValueError("MatMul Taint does not support 1D inputs yet")
  else:
    # Option 2: Batch MatMul (.., K, N), (.., N, M) -> (.., K, M)
    # print(f"MATMUL: {a.shape=} {b.shape=}")

    if jnp.shape(a)[:-2] != jnp.shape(b)[:-2]:
      a_shape = jnp.shape(a)
      b_shape = jnp.shape(b)

      max_ndim = max(a.ndim - 1, b.ndim - 1)
      if a.ndim - 1 < max_ndim:
        ndif = max_ndim - (a.ndim - 1)
        a = jnp.expand_dims(a, axis=list(range(ndif)))
        a = jnp.broadcast_to(a, b_shape[:ndif] + a_shape)
      elif b.ndim - 1 < max_ndim:
        ndif = max_ndim - (b.ndim - 1)
        b = jnp.expand_dims(b, axis=list(range(ndif)))
        b = jnp.broadcast_to(b, a_shape[:ndif] + b_shape)

    target_shape = jnp.shape(a)[:-1] + jnp.shape(b)[-1:]
    # print(f"MATMUL: {a.shape=} {b.shape=}    {target_shape=}")

    assert jnp.shape(a)[:-2] == jnp.shape(b)[:-2], "a and b must have the same shape[:-2]"
    assert jnp.shape(a)[-1] == jnp.shape(b)[-2], "a and b must have the same N"

    # reduce last dimension of a: (.., K, N) -> (.., K)
    a = taint_propagation.taint_reduction(a, dimensions=[a.ndim - 1])
    # reduce last-1 dimension of b: (.., N, M) -> (.., M)
    b = taint_propagation.taint_reduction(b, dimensions=[b.ndim - 2])

    # broadcast a to: (.., K) -> (.., K, M)
    a = jnp.reshape(a, jnp.shape(a) + (1,))
    a = jnp.broadcast_to(a, target_shape)

    # broadcast b to: (.., M) -> (.., K, M)
    b = jnp.reshape(b, jnp.shape(b)[:-1] + (1,) + (jnp.shape(b)[-1],))
    b = jnp.broadcast_to(b, target_shape)

    # elementwise taint:  {(.., K, M), (.., K, M)} -> (.., K, M)
    result = taint_propagation.binary_elementwise_taint(a, b)
    # print(f"MATMUL: {result}")
    return result


class TaintMatMul(matmul.MatMul, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {matmul.onnx_matmul: onnx_taint_matmul}

  @classmethod
  def data_handler(cls):
    return matmul.MatMul
