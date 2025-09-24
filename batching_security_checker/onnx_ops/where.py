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

"""Define ONNX Where operator."""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import where

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=())
def onnx_taint_where(*combined_args):
  """The internal jax impl for onnx Where op."""
  input_args, taint_input_args = taint_handler.split_args(combined_args, op="where")

  assert len(input_args) == 3
  assert len(taint_input_args) == 3
  cond, x, y = taint_input_args

  # NOTE: If cond is untainted (and deterministic),
  #  we can propagate taint more fine-grained.
  #       (using where.onnx_where)
  checkify.check(
      jnp.all(taint_propagation.is_determinisitic(cond)),
      "Op Where: The condition tensor cannot be partially random",
  )
  out = taint_propagation.binary_elementwise_taint(x, y)
  out = taint_propagation.binary_elementwise_taint(out, cond)
  return out


class TaintWhere(where.Where, taint_handler.TaintHandler):
  """Taint handler for Where operator."""

  @classmethod
  def jit_lookup(cls):
    return {where.onnx_where: onnx_taint_where}

  @classmethod
  def data_handler(cls):
    return where.Where
