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

"""Define ONNX ReduceMean operator."""

from jaxonnxruntime.onnx_ops import reducemean

from batching_security_checker.core import taint_handler
from batching_security_checker.onnx_ops import reducesum as reducesum_taint


class TaintReduceMean(reducemean.ReduceMean, taint_handler.TaintHandler):
  """Taint handler for ReduceMean operator."""

  @classmethod
  def jit_lookup(cls):
    # For taint propagation, we can use the same impl as ReduceSum.
    return {reducemean.onnx_reducemean: reducesum_taint.onnx_taint_reducesum}

  @classmethod
  def data_handler(cls):
    return reducemean.ReduceMean
