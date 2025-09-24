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

"""Define ONNX SkipLayerNormalization operator."""

import functools

import jax

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation
from batching_security_checker.onnx_ops_placeholders import skiplayernormalization

# pylint: disable=unused-argument


@functools.partial(jax.jit, static_argnames=("epsilon", "simplified"))
def onnx_taint_skiplayernormalization(*combined_args, epsilon, simplified):
  """The jax impl for taint propagation of onnx SkipLayerNormalization op."""

  _, taint_input_args = taint_handler.split_args(combined_args, op="skiplayernormalization")

  # inputs:     input, skip, gamma, [beta] [bias]
  if len(taint_input_args) == 3:
    data, skip, gamma = taint_input_args
    beta = None
    bias = None
    # print(f"len 3: {data.shape=} {skip=} {gamma=}")
  elif len(taint_input_args) == 4 and simplified:
    data, skip, gamma, bias = taint_input_args
    beta = None
    # print(f"len 4")
  elif len(taint_input_args) == 5:
    data, skip, gamma, beta, bias = taint_input_args
    # print(f"len 5: {data.shape=} {skip=} {gamma=} {beta=} {bias=}")
  else:
    raise ValueError(f"Unsupported number of inputs: {len(taint_input_args)}")

  val = taint_propagation.binary_elementwise_taint(data, skip)

  if bias is not None:
    val = taint_propagation.binary_elementwise_taint(val, bias)

  mean = taint_propagation.taint(val, axis=-1, keepdims=True)
  normalized = taint_propagation.binary_elementwise_taint(val, mean)

  # stage 2: scale + [bias] output
  output = taint_propagation.binary_elementwise_taint(normalized, gamma)

  if beta is not None:
    output = taint_propagation.binary_elementwise_taint(normalized, beta)

  mean = None
  inv_std_var = None
  input_skip_bias_sum = val
  return output, mean, inv_std_var, input_skip_bias_sum


class TaintSkipLayerNormalization(
    skiplayernormalization.SkipLayerNormalization, taint_handler.TaintHandler
):
  """Taint handler for SkipLayerNormalization operator."""

  @classmethod
  def jit_lookup(cls):
    return {
        skiplayernormalization.onnx_skiplayernormalization: (
            onnx_taint_skiplayernormalization
        )
    }

  @classmethod
  def data_handler(cls):
    return skiplayernormalization.SkipLayerNormalization
