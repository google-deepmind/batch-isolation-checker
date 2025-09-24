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


@handler.register_op("Gelu")
class Gelu(handler.Handler):
  """Implementation of the ONNX Gelu operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["approximate"] = node.attrs.get("approximate", "none")

  @classmethod
  def version_20(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_20 Gelu op."""
    cls._prepare(node, inputs, onnx_gelu)
    return onnx_gelu


@handler.register_op("Gelu", "com.microsoft")
class ContribGelu(handler.Handler):
  """Implementation of the ONNX Gelu operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    assert (
        "approximate" not in node.attrs_dict
    ), "approximate not supported in contrib gelu"
    node.attrs_dict["approximate"] = node.attrs.get("approximate", "none")

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 Gelu op."""
    cls._prepare(node, inputs, onnx_gelu)
    return onnx_gelu


@functools.partial(jax.jit, static_argnames=("approximate",))
def onnx_gelu(*input_args, approximate: Any):
  """https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gelu for more details."""

  assert len(input_args) == 1

  x = input_args[0]

  if approximate == "none":
    approximate = False
  elif approximate == "tanh":
    approximate = True
  else:
    raise ValueError(f"Unsupported approximate: {approximate}")

  output = jax.nn.gelu(x, approximate)
  return output
