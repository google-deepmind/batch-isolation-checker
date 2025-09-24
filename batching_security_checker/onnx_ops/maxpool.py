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

"""Define ONNX MaxPool operator."""

# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test

from collections.abc import Sequence
import functools
from typing import Union

import jax
from jax import lax
from jaxonnxruntime.onnx_ops import maxpool

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(
    jax.jit,
    static_argnames=(
        "ceil_mode",
        "strides",
        "pads",
        "dilations",
        "kernel_shape",
        "return_idx",
    ),
)
def onnx_taint_maxpool(
    *combined_args,
    ceil_mode: int,
    strides: Sequence[int],
    pads: Union[Sequence[tuple[int, int]], str],
    dilations: Sequence[int],
    kernel_shape: Sequence[int],
    storage_order: int,
    return_idx: bool = False,
):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#MaxPool for more details."""

  input_args, taint_input_args = taint_handler.split_args(combined_args, op="maxpool")

  assert len(input_args) == 1
  assert len(taint_input_args) == 1
  if return_idx:
    raise NotImplementedError("MaxPool with indices output is not implemented!")

  if ceil_mode != 0:
    raise ValueError("ceil_mode = 1 is not implement yet.")

  x = taint_input_args[0]
  identity = taint_propagation.identity_element()
  computation = taint_propagation.binary_elementwise_taint

  # NOTE: We do precise tainting here. Could also do more coarse as in the conv.
  pool_res = lax.reduce_window(
      x, identity, computation, kernel_shape, strides, pads, None, dilations
  )
  return pool_res


class TaintMaxPool(maxpool.MaxPool, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {maxpool.onnx_maxpool: onnx_taint_maxpool}

  @classmethod
  def data_handler(cls):
    return maxpool.MaxPool
