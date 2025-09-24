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

"""Define ONNX ConstantOfShape operator."""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import constantofshape

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=('value', 'shape', 'dtype'))
def onnx_taint_constantofshape(
    *combined_args, value=0, shape=None, dtype=jnp.float32  # pylint: disable=unused-argument
):
  """The internal jax impl for onnx ConstantOfShape op."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="constantofshape")

  assert len(input_args) == 1
  assert len(taint_input_args) == 1

  taint_shape = taint_input_args[0]

  checkify.check(
      taint_propagation.is_all_untainted_deterministic(taint_shape),
      'constantofshape: tainted or random input shape not supported',
  )
  return taint_propagation.identity_full(shape)


class TaintConstantOfShape(
    constantofshape.ConstantOfShape, taint_handler.TaintHandler
):

  @classmethod
  def jit_lookup(cls):
    return {constantofshape.onnx_constantofshape: onnx_taint_constantofshape}

  @classmethod
  def data_handler(cls):
    return constantofshape.ConstantOfShape
