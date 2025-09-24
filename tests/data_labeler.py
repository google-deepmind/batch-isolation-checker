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

"""Classes for generating random variations of label assignments for model inputs."""

import abc
import dataclasses
import random
from typing import Generator

from jax import typing as jtyping
import numpy as np
import onnx

from batching_security_checker.core import taint_propagation


@dataclasses.dataclass
class ModelIO:
  """Model inputs and outputs information."""

  model: onnx.ModelProto

  inputs_name: list[str]
  inputs: dict[str, np.ndarray]

  inputs_public: set[str]
  inputs_unchanged: set[str]

  outputs_name: list[str]
  outputs: dict[str, np.ndarray]


@dataclasses.dataclass
class InputsLabeling:
  """Label assignment for model inputs."""

  inputs_labeled: dict[
      str, np.ndarray
  ]  # NOTE: only contains the inputs that are labeled
  inputs_tainted: dict[str, jtyping.ArrayLike]  # NOTE: contains all inputs
  inputs_constants: dict[
      str, np.ndarray
  ]  # NOTE: for unlabeled inputs, contains the data constants

  labels: list[int]


@dataclasses.dataclass
class OutputsLabeling:
  """Label assignment for model outputs."""

  outputs_labeled: dict[str, jtyping.ArrayLike]

  labels: list[int]

  def labels_no_interference(self) -> set[int]:
    """Returns labels in any output without interference from other labels."""

    all_labels = set(self.labels)
    labels = set()
    for o in self.outputs_labeled.values():
      values, _ = np.unique(o, return_counts=True)
      for v in values:
        if v in all_labels:
          labels.add(v)

    return labels


class AbstractLabeler(abc.ABC):
  """Base class for generating label assignments for model inputs."""

  @abc.abstractmethod
  def generate(
      self,
      model_io: ModelIO,
  ) -> Generator[InputsLabeling, None, None]:
    pass


@dataclasses.dataclass
class DefaultLabeler(AbstractLabeler):
  """Assigns random labels to elements of all input tensors.

  Each element of the input tensors is assigned a random label.
  Inputs marked as "public" (c.f., ModelIO.inputs_public) are
  not labeled.

  Attributes:
      n_labelings: The number of different labelings to generate.
      max_n_labels: The maximum number of different labels to assign.
  """

  n_labelings: int = 20

  max_n_labels: int = 5

  def _sample_labels(
      self, name: str, model_io: ModelIO, n_labels: int
  ) -> np.ndarray:
    input_data = model_io.inputs[name]
    input_labeled = np.random.randint(0, n_labels, size=input_data.shape)
    return np.astype(input_labeled, np.uint32)

  def generate(
      self,
      model_io: ModelIO,
  ) -> Generator[InputsLabeling, None, None]:

    for _ in range(self.n_labelings):
      n_labels = random.randint(2, self.max_n_labels)

      inputs_labeled: dict[str, np.ndarray] = {}
      inputs_tainted: dict[str, jtyping.ArrayLike] = {}
      inputs_constants: dict[str, np.ndarray] = {}

      for name in model_io.inputs_name:
        input_data = model_io.inputs[name]

        is_labeled = name not in model_io.inputs_public

        assert isinstance(
            input_data, np.ndarray
        ), f'Only np.ndarray inputs are supported. {type(input_data)=}'

        if is_labeled:
          # randomly label the input and convert to taint
          input_labeled = self._sample_labels(name, model_io, n_labels)
          inputs_labeled[name] = input_labeled
          inputs_tainted[name] = taint_propagation.from_color_to_taint(
              input_labeled
          )
        else:  # is public input
          inputs_tainted[name] = np.full_like(
              input_data, taint_propagation.identity_element(), dtype=np.uint64
          )
          inputs_constants[name] = input_data

      yield InputsLabeling(
          inputs_labeled=inputs_labeled,
          inputs_tainted=inputs_tainted,
          inputs_constants=inputs_constants,
          labels=list(range(n_labels)),
      )


@dataclasses.dataclass
class MatMulLabeler(DefaultLabeler):
  """Custom labeler for matmul."""

  def _sample_labels(
      self, name: str, model_io: ModelIO, n_labels: int
  ) -> np.ndarray:
    """Creates a structured random labeling for input tensors of a matmul.

    In matmul, each output element is the result of a dot product between a row
    vector from the first matrix and a column vector from the second.
    Instead of assigning labels completely randomly, this function assigns a
    single random label to each such vector.

    This strategy increases the likelihood of having output elements without
    interference from multiple labels.

    This is accomplished by identifying the inner dimension of the matrix
    multiplication (the dimension along which the dot product is performed),
    sampling a random label for each vector along this dimension, and then
    broadcasting the label to the entire vector.

    Args:
        name: The name of the input tensor.
        model_io: The ModelIO object containing model information.
        n_labels: The number of labels to sample.

    Returns:
        A NumPy array containing the label tensor.
    """

    assert len(model_io.inputs_name) == 2, f'{len(model_io.inputs_name)=}'

    input_data = model_io.inputs[name]
    shape = list(input_data.shape)

    if name == model_io.inputs_name[0]:  # A input
      if len(shape) == 1:
        k_idx = 0
      else:
        k_idx = -1
    elif name == model_io.inputs_name[1]:  # B input
      if len(shape) == 1:
        k_idx = 0
      else:
        k_idx = -2
    else:
      raise ValueError(f'Unknown input name: {name}  {model_io.inputs_name=}')

    shape[k_idx] = 1
    input_labeled = np.random.randint(0, n_labels, size=shape)
    input_labeled = np.broadcast_to(input_labeled, shape=input_data.shape)
    return np.astype(input_labeled, np.uint32)


@dataclasses.dataclass
class GemmLabeler(DefaultLabeler):
  """Custom labeler for gemm."""

  def _sample_labels(
      self, name: str, model_io: ModelIO, n_labels: int
  ) -> np.ndarray:
    """Creates a structured random labeling for input tensors of gemm."""

    assert (
        len(model_io.inputs_name) == 2 or len(model_io.inputs_name) == 3
    ), f'{len(model_io.inputs_name)=}'

    input_data = model_io.inputs[name]
    shape = list(input_data.shape)

    def _get_int_attr(name: str):
      node = model_io.model.graph.node[0]
      for x in node.attribute:
        if x.name == name:
          return x.i
      return None

    if (
        len(model_io.inputs_name) == 3 and name == model_io.inputs_name[2]
    ):  # C input
      input_labeled = np.random.randint(0, n_labels, size=shape)
    else:
      if name == model_io.inputs_name[0]:  # A input
        k_idx = _get_int_attr('transA')
        if k_idx is None:
          k_idx = 1
        else:
          k_idx = int(not bool(k_idx))
      elif name == model_io.inputs_name[1]:  # B input
        k_idx = _get_int_attr('transB')
        if k_idx is None:
          k_idx = 0
      else:
        raise ValueError(f'Unknown input name: {name}  {model_io.inputs_name=}')
      shape[k_idx] = 1
      input_labeled = np.random.randint(0, n_labels, size=shape)
      input_labeled = np.broadcast_to(input_labeled, shape=input_data.shape)

    return np.astype(input_labeled, np.uint32)


@dataclasses.dataclass
class ConvLabeler(DefaultLabeler):
  """Custom labeler for convolution."""

  def _sample_labels(
      self, name: str, model_io: ModelIO, n_labels: int
  ) -> np.ndarray:
    """Creates a structured random labeling for input tensors of a convolution.

    The data input receives a single random label for each batch position.
    The kernel input receives a single label.

    Args:
        name: The name of the input tensor.
        model_io: The ModelIO object containing model information.
        n_labels: The number of labels to sample.

    Returns:
        A NumPy array containing the label tensor.
    """

    assert (
        len(model_io.inputs_name) == 3 or len(model_io.inputs_name) == 2
    ), f'{len(model_io.inputs_name)=}'

    input_data = model_io.inputs[name]

    if name == model_io.inputs_name[0]:  # X
      # Choose a random label per batch position
      # set everything except batch size dim (0th index) to 1
      shape = list(input_data.shape)
      for i in range(1, len(shape)):
        shape[i] = 1
      input_labeled = np.random.randint(0, n_labels, size=shape)

    elif name == model_io.inputs_name[1]:  # Kernel
      # Choose a single random label for the kernel
      shape = list(input_data.shape)
      for i in range(0, len(shape)):
        shape[i] = 1
      input_labeled = np.random.randint(0, n_labels, size=shape)
    else:
      raise ValueError(f'Unknown input name: {name}  {model_io.inputs_name=}')

    input_labeled = np.broadcast_to(input_labeled, shape=input_data.shape)
    return np.astype(input_labeled, np.uint32)


@dataclasses.dataclass
class MaxPoolLabeler(DefaultLabeler):
  """Custom labeler for maxpool."""

  def _sample_labels(
      self, name: str, model_io: ModelIO, n_labels: int
  ) -> np.ndarray:
    """Creates a structured random labeling for input tensors of a maxpool.

    Args:
        name: The name of the input tensor.
        model_io: The ModelIO object containing model information.
        n_labels: The number of labels to sample.

    Returns:
        A NumPy array containing the label tensor.
    """

    assert len(model_io.inputs_name) == 1, f'{len(model_io.inputs_name)=}'

    input_data = model_io.inputs[name]

    # Choose a random label per batch position
    # set everything except batch size dim (0th index) to 1
    # TODO: Could be made more fine-grained
    shape = list(input_data.shape)
    for i in range(1, len(shape)):
      shape[i] = 1
    input_labeled = np.random.randint(0, n_labels, size=shape)

    input_labeled = np.broadcast_to(input_labeled, shape=input_data.shape)
    return np.astype(input_labeled, np.uint32)
