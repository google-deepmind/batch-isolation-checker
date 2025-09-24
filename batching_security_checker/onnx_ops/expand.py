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

"""Define ONNX Expand operator."""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import expand

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames="shape")
def onnx_taint_expand(*combined_args, shape):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Expand for more details."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="expand")

  assert len(input_args) == 2
  assert len(taint_input_args) == 2

  taint_data, taint_shape = taint_input_args

  checkify.check(
      taint_propagation.is_all_untainted_deterministic(taint_shape),
      "expand: tainted or random input shape not supported",
  )

  # expands the taint by broadcasting
  return taint_data * jnp.ones(shape, dtype=taint_data.dtype)


class TaintExpand(expand.Expand, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {expand.onnx_expand: onnx_taint_expand}

  @classmethod
  def data_handler(cls):
    return expand.Expand
