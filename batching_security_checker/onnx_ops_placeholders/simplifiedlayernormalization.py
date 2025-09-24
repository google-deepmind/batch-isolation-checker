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

"""Define ONNX SimplifiedLayerNormalization operator."""

from collections.abc import Callable, Sequence
from typing import Any

from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node

from batching_security_checker.onnx_ops_placeholders import layernormalization


@handler.register_op("SimplifiedLayerNormalization", "experimental")
class SimplifiedLayerNormalization(handler.Handler):
  """Implementation of the ONNX SimplifiedLayerNormalization operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict["axis"] = node.attrs.get("axis", -1)
    node.attrs_dict["epsilon"] = node.attrs.get("epsilon", 1e-05)
    node.attrs_dict["stash_type"] = node.attrs.get("stash_type", 1)

    # NOTE: We use implementation of LayerNormalization, but we set
    # simplified to 1.
    node.attrs_dict["simplified"] = 1

  @classmethod
  def version_1(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_1 SimplifiedLayerNormalization op."""
    cls._prepare(node, inputs, layernormalization.onnx_layernormalization)
    return layernormalization.onnx_layernormalization
