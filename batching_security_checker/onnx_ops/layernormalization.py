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

"""Define ONNX LayerNormalization operator."""

import functools

import jax

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation
from batching_security_checker.onnx_ops_placeholders import layernormalization


@functools.partial(
    jax.jit, static_argnames=("axis", "epsilon", "stash_type", "simplified")
)
def onnx_taint_layernormalization(
    *combined_args, axis, epsilon, stash_type, simplified  # pylint: disable=unused-argument
):
  """The jax impl for taint propagation of onnx LayerNormalization op."""

  _, taint_input_args = taint_handler.split_args(combined_args, op="layernormalization")

  if len(taint_input_args) == 2:
    data, scale = taint_input_args
    bias = None
  elif len(taint_input_args) == 3:
    data, scale, bias = taint_input_args
  else:
    raise ValueError(f"Unsupported number of inputs: {len(taint_input_args)}")

  # stage 1: taint propagation corresponding to normalize
  mean = taint_propagation.taint(data, axis=axis, keepdims=True)
  normalized = taint_propagation.binary_elementwise_taint(data, mean)

  # stage 2: taint propagation corresponding to scale
  result = taint_propagation.binary_elementwise_taint(normalized, scale)

  if bias:
    result = taint_propagation.binary_elementwise_taint(result, bias)

  return result


class TaintLayerNormalization(
    layernormalization.LayerNormalization, taint_handler.TaintHandler
):
  """Taint handler for LayerNormalization operator."""

  @classmethod
  def jit_lookup(cls):
    return {
        layernormalization.onnx_layernormalization: (
            onnx_taint_layernormalization
        )
    }

  @classmethod
  def data_handler(cls):
    return layernormalization.LayerNormalization
