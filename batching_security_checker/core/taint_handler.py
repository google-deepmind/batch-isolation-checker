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

"""Base logic for taint handlers."""

from collections.abc import Callable, Sequence
import logging
from typing import Any, Dict, Tuple, Type

from jax import numpy as jnp

from jax.experimental import checkify
from jaxonnxruntime.core import handler as onnx_handler
from jaxonnxruntime.core import onnx_node


OnnxNode = onnx_node.OnnxNode

logger = logging.getLogger(__name__)


def split_args(combined_args, op=None) -> tuple[Sequence[Any], Sequence[Any]]:
  """Splits the combined_args into input_args and taint_input_args."""

  assert len(combined_args) % 2 == 0, f"combined_args must be even length: len={len(combined_args)}  ({op=})"

  input_args = combined_args[: len(combined_args) // 2]
  taint_input_args = combined_args[len(combined_args) // 2 :]

  # perform sanity checks
  for input_arg, taint_arg in zip(input_args, taint_input_args, strict=True):
    if input_arg is None:
      checkify.check(
          taint_arg is None,
          "taint_arg must be None if input_arg is None ({taint_arg=})",
      )
    else:
      checkify.check(
          jnp.shape(input_arg) == jnp.shape(taint_arg),
          "{input_arg=} and {taint_arg=} must have the same shape.",
      )

      # NOTE: The sanity check is deactivated because
      #       it doesn't work for non-floating point tensors.
      # is_untainted = taint_propagation.is_all_untainted(taint_arg)
      # is_data_nan = jnp.all(jnp.isnan(input_arg))
      # checkify.check(
      #    jnp.logical_or(is_untainted, is_data_nan),
      #    "A tainted input must be nan for data: is_untainted={is_untainted} "
      #    " is_data_nan={is_data_nan}",
      #    is_untainted=is_untainted,
      #    is_data_nan=is_data_nan,
      # )

  return input_args, taint_input_args


class TaintHandler:
  """Base class for taint handlers."""

  @classmethod
  def prepare_taint_attrs(cls, attrs_dict: dict[str, Any]) -> dict[str, Any]:
    return attrs_dict

  @classmethod
  def jit_lookup(cls) -> Dict[Callable[..., Any], Callable[..., Any]]:
    """A class can override this method to provide a dictionary of taint functions."""
    raise ValueError(f"No jit_lookup for {cls.__name__}. Please implement.")

  @classmethod
  def data_handler(cls) -> Type[onnx_handler.Handler]:
    raise ValueError(f"No data handler for {cls.__name__}. Please implement.")

  @classmethod
  def handle_taint(
      cls,
      node: OnnxNode,
      data_inputs: Sequence[Any],
      taint_inputs: Sequence[Any],
      **kwargs,
  ) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Returns the data and taint jit functions for the node."""

    assert len(data_inputs) == len(
        taint_inputs
    ), "inputs and tainted_inputs must be the same length"

    # determine op version, setup node.attrs_dict and obtain jit func
    data_jit_func = cls.data_handler().handle(node, data_inputs, **kwargs)

    jit_lookup = cls.jit_lookup()

    if data_jit_func not in jit_lookup:
      raise ValueError(f"No taint func found for {data_jit_func}.")
    taint_jit_func = jit_lookup[data_jit_func]

    # NOTE: Logic to determine if all inputs are useable or not.
    #       -> if it contains private inputs, then we set data_jit_func to None.
    has_all_inputs_available = True
    if not has_all_inputs_available:
      data_jit_func = None

    # # check that node.attrs_dict does not contain any nans
    # for k, v in node.attrs_dict.items():
    #   print(f"k: {k}, v: {v}  {type(v)}")
    #   if v:
    #     if isinstance(v, tuple) or isinstance(v, str):
    #       continue

    #     has_nan = jnp.any(jnp.isnan(v))
    #     #assert has_nan.size() == 1
    #     if bool(has_nan):
    #       raise ValueError(f"Node {node.name} has nan in attrs_dict: {k}.")

    return data_jit_func, taint_jit_func
