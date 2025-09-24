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

from batching_security_checker.core import taint_handler
from batching_security_checker.onnx_ops import skiplayernormalization as taint_ssln
from batching_security_checker.onnx_ops_placeholders import skiplayernormalization
from batching_security_checker.onnx_ops_placeholders import skipsimplifiedlayernormalization


class TaintSkipSimplifiedLayerNormalization(
    skipsimplifiedlayernormalization.SkipSimplifiedLayerNormalization,
    taint_handler.TaintHandler,
):
  """Taint handler for SkipSimplifiedLayerNormalization operator."""

  @classmethod
  def jit_lookup(cls):
    return {
        skiplayernormalization.onnx_skiplayernormalization: (
            taint_ssln.onnx_taint_skiplayernormalization
        )
    }

  @classmethod
  def data_handler(cls):
    return skipsimplifiedlayernormalization.SkipSimplifiedLayerNormalization
