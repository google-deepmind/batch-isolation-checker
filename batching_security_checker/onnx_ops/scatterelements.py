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

"""Define ONNX ScatterElements operator."""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import scatterelements

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=("axis", "reduction"))
def onnx_taint_scatterelements(*combined_args, axis=0, reduction=None):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#ScatterElements."""
  input_args, taint_input_args = taint_handler.split_args(combined_args, op="scatterelements")

  assert len(input_args) == 3
  assert len(taint_input_args) == 3
  _, indices, _ = input_args

  taint_data, taint_indices, taint_updates = taint_input_args

  checkify.check(
      taint_propagation.is_all_untainted_deterministic(taint_indices),
      "scatterelements: tainted or random indices not supported",
  )

  if jnp.shape(taint_indices) != jnp.shape(taint_updates):
    raise ValueError(
        "taint_indices and taint_updates must have the same shape."
        f" ({jnp.shape(taint_indices)=} != {jnp.shape(taint_updates)=})"
    )

  # propagate the taint from taint_data
  result = scatterelements.onnx_scatterelements(
      *[taint_data, indices, taint_updates], axis=axis, reduction=None
  )

  if jnp.shape(taint_data) != jnp.shape(result):
    raise ValueError(
        "taint_data and result must have the same shape."
        f" ({jnp.shape(taint_data)=} != {jnp.shape(result)=})"
    )

  if reduction is None:
    pass
  elif reduction == "add" or reduction == "mul":
    result = taint_propagation.binary_elementwise_taint(result, taint_data)
  else:
    raise ValueError(f"{reduction=} not yet supported in taint propagation.")
  return result


class TaintScatterElements(
    scatterelements.ScatterElements, taint_handler.TaintHandler
):

  @classmethod
  def jit_lookup(cls):
    return {scatterelements.onnx_scatterelements: onnx_taint_scatterelements}

  @classmethod
  def data_handler(cls):
    return scatterelements.ScatterElements
