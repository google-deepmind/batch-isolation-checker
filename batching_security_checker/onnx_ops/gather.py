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

"""Define ONNX Gather operator."""

import functools

import jax
from jax import numpy as jnp
from jaxonnxruntime.onnx_ops import gather

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames="axis")
def onnx_taint_gather(*combined_args, axis=0):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Gather."""
  input_args, taint_input_args = taint_handler.split_args(combined_args, op="gather")

  assert len(input_args) == 2
  assert len(taint_input_args) == 2
  _, indices = input_args

  taint_data, taint_indices = taint_input_args

  def untainted_deterministic_indices_gather(
      taint_data, taint_indices, indices  # pylint: disable=unused-argument
  ):
    indices = indices.astype(jnp.int64)
    out = jnp.take(taint_data, indices, axis=axis)
    return out

  def tainted_indices_gather(taint_data, taint_indices, indices):  # pylint: disable=unused-argument

    # 1. reduce data taint
    data_reduced = taint_propagation.taint_reduction(
        taint_data, dimensions=[axis]
    )
    # 2. expand reduced data taint

    # expand dims so that it now has same number of dims as output shape
    #                (1, 1, 1, S1, S2, S3, S4) for axis=0
    #                (S1, 1, 1, 1, S2, S3, S4) for axis=1
    #                (S1, S2, 1, 1, 1, S3, S4) for axis=2
    axes = tuple(map(lambda x: x + axis, range(taint_indices.ndim)))
    data_reduced = jnp.expand_dims(data_reduced, axis=axes)

    # 3. expand taint_indices (axes that are not already in data_reduced)
    axes = tuple(sorted(set(range(data_reduced.ndim)).difference(set(axes))))
    taint_indices = jnp.expand_dims(taint_indices, axis=axes)
    assert taint_indices.ndim == data_reduced.ndim

    # data_reduced.shape =  ( 1,  1,  1,  S1, S2, S3, S4) # for axis=0
    # taint_indices.shape = (I1, I2, I3,   1,  1,  1,  1)

    # data_reduced.shape =  ( S1,  1,  1,  1, S2, S3, S4) # for axis=1
    # taint_indices.shape = (1, I1, I2, I3,   1,  1,  1)

    # data_reduced.shape =  ( S1,  S2,  1,  1, 1, S3, S4) # for axis=2
    # taint_indices.shape = (1, 1, I1, I2, I3, 1,  1)

    # 4. binary elementwise taint of both (with broadcasting)
    out = taint_propagation.binary_elementwise_taint(
        data_reduced, taint_indices
    )

    return out

  out = jax.lax.cond(
      jnp.logical_and(
          taint_propagation.is_all_untainted(taint_indices),
          taint_propagation.is_all_deterministic(taint_indices),
      ),
      untainted_deterministic_indices_gather,
      tainted_indices_gather,
      taint_data,
      taint_indices,
      indices,
  )

  return out


class TaintGather(gather.Gather, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {gather.onnx_gather: onnx_taint_gather}

  @classmethod
  def data_handler(cls):
    return gather.Gather
