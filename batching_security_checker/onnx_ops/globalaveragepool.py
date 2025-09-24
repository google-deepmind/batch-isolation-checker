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

"""Define ONNX GlobalAveragePool operator."""

import functools

import jax
from jax import numpy as jnp
from jaxonnxruntime.onnx_ops import globalaveragepool

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(
    jax.jit,
    static_argnames=(),
)
def onnx_taint_globalaveragepool(
    *combined_args,
):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#GlobalAveragePool for more details."""

  _, taint_input_args = taint_handler.split_args(combined_args, op="globalaveragepool")

  assert len(taint_input_args) == 1

  x = taint_input_args[0]

  out = taint_propagation.taint_reduction(
      x, dimensions=tuple(range(2, jnp.ndim(x)))
  )

  shape = jnp.shape(out) + (1, 1)
  out = jnp.reshape(out, shape)

  return out


class TaintGlobalAveragePool(
    globalaveragepool.GlobalAveragePool, taint_handler.TaintHandler
):
  """Global average pool operator taint handler."""

  @classmethod
  def jit_lookup(cls):
    return {
        globalaveragepool.onnx_globalaveragepool: onnx_taint_globalaveragepool
    }

  @classmethod
  def data_handler(cls):
    return globalaveragepool.GlobalAveragePool
