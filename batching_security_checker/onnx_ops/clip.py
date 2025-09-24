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

"""Define ONNX Clip operator."""

import functools

import jax
from jaxonnxruntime.onnx_ops import clip

from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation


@functools.partial(jax.jit, static_argnames=())
def onnx_taint_clip(*combined_args):  #, amin=None, amax=None # TODO: These args are optional but caused problems, by removing them also from the data version it worked and is correct for taint analysis
  """The internal jax impl for onnx Clip op."""
  input_args, taint_input_args = taint_handler.split_args(combined_args, op="clip")

  if len(input_args) == 3:
    assert len(taint_input_args) == 3
    x, minimum, maximum = taint_input_args
  elif len(input_args) == 2:
    assert len(taint_input_args) == 2
    x, minimum = taint_input_args
    maximum = None
  elif len(input_args) == 1:
    x = taint_input_args
    minimum = None # controlled via attributes
    maximum = None
  else:
    raise ValueError(f"Unexpected number of arguments: {len(input_args)}")

  if minimum is not None:
    x = taint_propagation.binary_elementwise_taint(x, minimum)

  if maximum is not None:
    x = taint_propagation.binary_elementwise_taint(x, maximum)
  return x


class TaintClip(clip.Clip, taint_handler.TaintHandler):

  @classmethod
  def jit_lookup(cls):
    return {clip.onnx_clip: onnx_taint_clip}

  @classmethod
  def data_handler(cls):
    return clip.Clip
