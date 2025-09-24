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

"""Define ONNX Unsqueeze operator."""

import functools

import jax
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import unsqueeze

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames="axis")
def onnx_taint_unsqueeze(*combined_args, axis: list[int]):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Unsqueeze."""

  _, taint_input_args = taint_handler.split_args(combined_args, op="unsqueeze")

  assert len(taint_input_args) == 1 or len(taint_input_args) == 2

  if len(taint_input_args) >= 2:
    _, axes_taint = taint_input_args

    checkify.check(
        taint_propagation.is_all_untainted_deterministic(axes_taint),
        "unsqueeze: tainted input axes not supported",
    )

  result = unsqueeze.onnx_unsqueeze(*taint_input_args, axis=axis)

  return result


class TaintUnsqueeze(unsqueeze.Unsqueeze, taint_handler.TaintHandler):
  """Taint handler for Unsqueeze operator."""

  @classmethod
  def jit_lookup(cls):
    return {unsqueeze.onnx_unsqueeze: onnx_taint_unsqueeze}

  @classmethod
  def data_handler(cls):
    return unsqueeze.Unsqueeze
