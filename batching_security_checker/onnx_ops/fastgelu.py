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

"""Define ONNX Gelu operator.

(There is a version in the com.microsoft domain and in the default domain)
"""

import functools

import jax

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation
from batching_security_checker.onnx_ops_placeholders import fastgelu


@functools.partial(jax.jit, static_argnames=())
def onnx_taint_fastgelu(*combined_args):
  """https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FastGelu for more details."""

  _, taint_input_args = taint_handler.split_args(combined_args, op="fastgelu")

  if len(taint_input_args) == 1:
    x = taint_input_args[0]
  elif len(taint_input_args) == 2:
    x, bias = taint_input_args
    x = taint_propagation.binary_elementwise_taint(x, bias)
  else:
    raise ValueError(f"Unsupported number of inputs: {len(taint_input_args)}")

  return taint_propagation.identity_taint(x)


class TaintFastGelu(fastgelu.FastGelu, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {fastgelu.onnx_fastgelu: onnx_taint_fastgelu}

  @classmethod
  def data_handler(cls):
    return fastgelu.FastGelu
