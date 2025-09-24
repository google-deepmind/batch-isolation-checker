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

"""Define ONNX Transpose operator."""

import functools

import jax
from jaxonnxruntime.onnx_ops import transpose

from batching_security_checker.core import taint_handler


@functools.partial(jax.jit, static_argnames="perm")
def onnx_taint_transpose(*combined_args, perm):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Transpose."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="transpose")

  assert len(input_args) == 1
  assert len(taint_input_args) == 1

  # We call original transpose function on the taint input
  result = transpose.onnx_transpose(*taint_input_args, perm=perm)
  return result


class TaintTranspose(transpose.Transpose, taint_handler.TaintHandler):
  """Taint handler for Transpose operator."""

  @classmethod
  def jit_lookup(cls):
    return {transpose.onnx_transpose: onnx_taint_transpose}

  @classmethod
  def data_handler(cls):
    return transpose.Transpose
