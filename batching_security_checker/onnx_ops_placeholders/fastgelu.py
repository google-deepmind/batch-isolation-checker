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

from collections.abc import Callable, Sequence
import functools
from typing import Any

import jax
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node


@handler.register_op("FastGelu", "com.microsoft")
class FastGelu(handler.Handler):
  """Implementation of the ONNX FastGelu operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    pass

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 FastGelu op."""
    cls._prepare(node, inputs, onnx_fastgelu)
    return onnx_fastgelu


@functools.partial(jax.jit, static_argnames=())
def onnx_fastgelu(*input_args):
  """https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FastGelu for more details."""

  if len(input_args) == 1:
    x = input_args[0]
  elif len(input_args) == 2:
    x, bias = input_args
    x = x + bias
  else:
    raise ValueError(f"Unsupported number of inputs: {len(input_args)}")

  output = jax.nn.gelu(x, approximate=True)
  return output
