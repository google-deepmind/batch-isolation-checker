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

"""Define ONNX ScatterND operator."""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import scatternd

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames="reduction")
def onnx_taint_scatternd(*combined_args, reduction=None):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ScatterND."""
  input_args, taint_input_args = taint_handler.split_args(combined_args, op="scatternd")

  assert len(input_args) == 3
  assert len(taint_input_args) == 3
  _, indices, _ = input_args

  taint_data, taint_indices, taint_updates = taint_input_args

  checkify.check(
      taint_propagation.is_all_untainted_deterministic(taint_indices),
      "scatternd: tainted or random indices not supported",
  )

  # propagate the taint from taint_data
  result = scatternd.onnx_scatternd(
      *[taint_data, indices, taint_updates], reduction="set"
  )

  if jnp.shape(taint_data) != jnp.shape(result):
    raise ValueError(
        "taint_data and result must have the same shape."
        f" ({jnp.shape(taint_data)=} != {jnp.shape(result)=})"
    )

  if reduction is None or reduction == "set":
    pass
  elif reduction in ["add", "multiply", "max", "min"]:
    result = taint_propagation.binary_elementwise_taint(result, taint_data)
  else:
    raise ValueError(f"{reduction=} not yet supported in taint propagation.")
  return result


class TaintScatterND(scatternd.ScatterND, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {scatternd.onnx_scatternd: onnx_taint_scatternd}

  @classmethod
  def data_handler(cls):
    return scatternd.ScatterND
