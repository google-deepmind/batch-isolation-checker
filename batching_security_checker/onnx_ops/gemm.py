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

"""Define ONNX Gemm operator."""

import functools
from typing import Optional

import jax
from jax import numpy as jnp
from jaxonnxruntime.onnx_ops import gemm

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation
from batching_security_checker.onnx_ops import matmul


@functools.partial(
    jax.jit, static_argnames=('alpha', 'beta', 'transA', 'transB')
)
def onnx_taint_gemm(
    *combined_args,
    alpha: Optional[float] = None,  # pylint: disable=unused-argument
    beta: Optional[float] = None,  # pylint: disable=unused-argument
    transA: Optional[int] = None,
    transB: Optional[int] = None,
):
  """The impl for https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Gemm."""

  _, taint_input_args = taint_handler.split_args(combined_args, op="gemm")

  assert len(taint_input_args) == 3 or len(taint_input_args) == 2
  if len(taint_input_args) == 2:
    a, b = taint_input_args
    c = None
  else:
    a, b, c = taint_input_args

  transA = 0 if not transA else transA
  transB = 0 if not transB else transB

  if transA == 1:
    a = jnp.transpose(a)
  if transB == 1:
    b = jnp.transpose(b)

  out = matmul.onnx_taint_matmul(a, b, a, b)

  if c is not None:
    out = taint_propagation.binary_elementwise_taint(out, c)
  return out


class TaintGemm(gemm.Gemm, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {gemm.onnx_gemm: onnx_taint_gemm}

  @classmethod
  def data_handler(cls):
    return gemm.Gemm
