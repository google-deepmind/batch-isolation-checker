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

"""Define ONNX BatchNormalization operator."""

import functools

import jax
from jax import numpy as jnp
from jaxonnxruntime.onnx_ops import batchnormalization

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(
  jax.jit,
  static_argnames=('epsilon', 'momentum', 'training_mode')
)
def onnx_taint_batchnormalization(*combined_args, epsilon: float, momentum: float, training_mode: int):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#BatchNormalization."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="batchnormalization")

  if len(input_args) == 5 and len(taint_input_args) == 5:
    x, scale, b, input_mean, input_var = taint_input_args
  else:
    raise ValueError(f"Unknown input size: {len(input_args)=}")

  dims_x = jnp.ndim(x)
  dim_ones = (1,) * (dims_x - 2)
  scale = scale.reshape(-1, *dim_ones)
  b = b.reshape(-1, *dim_ones)
  input_mean = input_mean.reshape(-1, *dim_ones)
  input_var = input_var.reshape(-1, *dim_ones)

  if training_mode == 0:
    out = taint_propagation.binary_elementwise_taint(x, input_mean)
    out = taint_propagation.binary_elementwise_taint(out, input_var)
    out = taint_propagation.binary_elementwise_taint(out, scale)
    out = taint_propagation.binary_elementwise_taint(out, b)
    return out
  else:
    raise NotImplementedError(
        'BatchNormalization with training_mode was not implemented yet.'
    )


class TaintBatchNormalization(batchnormalization.BatchNormalization, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {batchnormalization.onnx_batchnormalization: onnx_taint_batchnormalization}

  @classmethod
  def data_handler(cls):
    return batchnormalization.BatchNormalization
