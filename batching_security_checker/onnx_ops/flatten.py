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

"""Define ONNX Flatten operator."""

import functools

import jax
from jaxonnxruntime.onnx_ops import flatten

from batching_security_checker.core import taint_handler


@functools.partial(jax.jit, static_argnames="axis")
def onnx_taint_flatten(*combined_args, axis):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Flatten."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="flatten")

  assert len(input_args) == 1
  assert len(taint_input_args) == 1

  # We call original flatten function on the taint input
  result = flatten.onnx_flatten(*taint_input_args, axis=axis)
  return result


class TaintFlatten(flatten.Flatten, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {flatten.onnx_flatten: onnx_taint_flatten}

  @classmethod
  def data_handler(cls):
    return flatten.Flatten
