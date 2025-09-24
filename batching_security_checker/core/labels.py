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

"""Helper functions for creating labeled inputs and outputs for a model."""

import abc
from typing import Sequence, Tuple, Union

import attrs as attr
from jax import numpy as jnp
import onnx

from batching_security_checker.core import model_summary
from batching_security_checker.core import taint_propagation


@attr.define
class AbstractLabeler(abc.ABC):
  """Base class for generating label assignments for model inputs."""

  @abc.abstractmethod
  def get_labeled_inputs_outputs(
      self,
      model: onnx.ModelProto,
      dim_params: dict[str, int],
  ) -> Tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    pass


@attr.define
class BatchDimLabeler(AbstractLabeler):
  """Assigns labels to inputs and outputs based on batch dimension.

    The assumption is that in every input and output tensor, a dimension is used
    as the batch dimension. By default, `io_batch_dim` is used as the default
    batch dimension for every input and output tensor. If the batch dimension
    is a dynamic parameter, it's also possible to choose the batch dimension by
    name. By providing a dictionary of exceptions, the batch dimension can be
    set for specific tensors.

    To generate concrete inputs and outputs, the dynamic parameters are replaced
    with the values provided in `dim_params`.

    Args:
      model: The ONNX model.
      io_batch_dim: The default batch dimension for input and output tensors. It
        can be either an integer representing the dimension index or a string
        representing the dynamic dimension name.
      io_batch_dim_exceptions: A dictionary mapping tensor names to their
        specific batch dimensions. This allows overriding the default
        `io_batch_dim` for certain tensors. Keys are tensor names (strings), and
        values must be integers (dimension indices).
      dim_params: A dictionary mapping dynamic dimension names to their concrete
        values. Keys are dimension names (strings), and values are integers. The
        names must match the names of the dynamic dimensions in the model.
      label_dtype: The data type for the labels. Must be one of `np.uint8`,
        `np.uint16`, or `np.uint32`.

    Returns:
      A tuple containing two dictionaries:
        - labeled_inputs: A dictionary mapping input tensor names to their
          corresponding labeled input tensors (NumPy arrays).
        - labeled_outputs: A dictionary mapping output tensor names to their
          corresponding labeled output tensors (NumPy arrays).
  """

  io_batch_dim: Union[int, str]
  io_batch_dim_exceptions: dict[str, int] = attr.field(
      default=attr.Factory(dict)
  )
  label_dtype: Union[type[jnp.uint8], type[jnp.uint16], type[jnp.uint32]] = (
      attr.field(
          metadata={"exclude": True},
          default=attr.Factory(lambda: taint_propagation.tdtype().jnp_part)
      )
  )

  def get_labeled_inputs_outputs(
      self,
      model: onnx.ModelProto,
      dim_params: dict[str, int],
  ) -> Tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:

    dim_param_names, inputs, outputs = model_summary.model_inputs_outputs(model)

    if set(dim_param_names) != set(dim_params.keys()):
      missing = set(dim_param_names) - set(dim_params.keys())
      extra = set(dim_params.keys()) - set(dim_param_names)
      raise ValueError(
          f"Provided dim_params do not match model: {missing=}  {extra=}"
      )

    #if not all(
    #    x in dim_param_names for x in self.io_batch_dim_exceptions.keys()
    #):
    #  raise ValueError(f"Batch dim exceptions do not match model: {dim_param_names=}     {self.io_batch_dim_exceptions=}")

    if (
        isinstance(self.io_batch_dim, str)
        and self.io_batch_dim not in dim_param_names
    ):
      raise ValueError(
          f"Batch dim {self.io_batch_dim} is not a dynamic model param"
      )

    def _create_labeled_tensors(
        tensors: dict[str, Sequence[Union[int, str]]],
    ) -> Tuple[dict[str, jnp.ndarray], int]:
      labeled_tensors: dict[str, jnp.ndarray] = {}
      batch_size = None
      for name, shape in tensors.items():

        if isinstance(self.io_batch_dim, str):
          default_dim = shape.index(self.io_batch_dim)
        elif isinstance(self.io_batch_dim, int):
          default_dim = self.io_batch_dim
        else:
          raise ValueError(f"Unknown batch dim type: {self.io_batch_dim}")

        shape_fixed = _replace_dim_params(shape, dim_params)

        dim = self.io_batch_dim_exceptions.get(name, default_dim)

        if batch_size is None:
          batch_size = shape_fixed[dim]
        elif batch_size != shape_fixed[dim]:
          raise ValueError(
              f"Inconsistent batch size: {batch_size=} {shape_fixed[dim]=}"
          )
        del shape_fixed[dim]

        parts = []
        for label in range(1, batch_size + 1):
          x = jnp.full(shape_fixed, label, dtype=self.label_dtype)
          parts.append(x)

        labeled_tensors[name] = jnp.stack(parts, axis=dim)

      if batch_size is None:
        raise ValueError("No tensors provided")
      return labeled_tensors, batch_size

    labeled_inputs, batch_size_inputs = _create_labeled_tensors(tensors=inputs)
    labeled_outputs, batch_size_outputs = _create_labeled_tensors(
        tensors=outputs
    )

    assert (
        batch_size_inputs == batch_size_outputs
    ), f"Inconsistent batch size: {batch_size_inputs=} != {batch_size_outputs=}"

    return labeled_inputs, labeled_outputs


# NOTE: copy from analysis code
def _replace_dim_params(
    shape: Sequence[Union[int, str]], dim_param_values: dict[str, int]
):
  new_shape = []
  for s in shape:
    if isinstance(s, str):
      new_shape.append(dim_param_values[s])
    elif isinstance(s, int):
      new_shape.append(s)
    else:
      raise ValueError(f"Unknown dim type: {s}")
  return new_shape
