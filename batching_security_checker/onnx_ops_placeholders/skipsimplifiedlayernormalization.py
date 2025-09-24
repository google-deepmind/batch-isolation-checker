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

from collections.abc import Callable, Sequence
from typing import Any

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node

from batching_security_checker.onnx_ops_placeholders import skiplayernormalization


@handler.register_op("SkipSimplifiedLayerNormalization", "com.microsoft")
class SkipSimplifiedLayerNormalization(handler.Handler):
  """Implementation of the ONNX SkipSimplifiedLayerNormalization operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):

    node.attrs_dict["epsilon"] = node.attrs.get("epsilon", 1e-05)
    node.attrs_dict["simplified"] = True

  # TODO: What are versions?
  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 SkipSimplifiedLayerNormalization op."""
    cls._prepare(
        node, inputs, skiplayernormalization.onnx_skiplayernormalization
    )
    return skiplayernormalization.onnx_skiplayernormalization
