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

"""Define ONNX Trilu operator."""

import functools

import jax
from jax import numpy as jnp
from jax.experimental import checkify
from jaxonnxruntime.onnx_ops import trilu

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=("upper", "k"))
def onnx_taint_trilu(*combined_args, k, upper):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Trilu."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="trilu")

  if len(input_args) == 2:
    assert len(taint_input_args) == 2
    _, taint_k = taint_input_args
  elif len(input_args) == 1:
    assert len(taint_input_args) == 1
    taint_k = None
  else:
    raise ValueError(f"Unexpected number of arguments: {len(input_args)}")

  if taint_k is not None:
    checkify.check(
        taint_propagation.is_all_untainted_deterministic(taint_k),
        "trilu: tainted k not supported",
    )

  # We call original trilu function but ensure that k is untainted.
  result = trilu.onnx_trilu(*taint_input_args, upper=upper, k=k)
  # The trilu op sets some elements to 0, so we need to set those to untainted.
  result = jnp.where(result == 0, taint_propagation.identity_element(), result)
  return result


class TaintTrilu(trilu.Trilu, taint_handler.TaintHandler):
  """Taint handler for Trilu operator."""

  @classmethod
  def jit_lookup(cls):
    return {trilu.onnx_trilu: onnx_taint_trilu}

  @classmethod
  def data_handler(cls):
    return trilu.Trilu
