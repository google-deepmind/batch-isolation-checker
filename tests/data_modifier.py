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

"""Classes for generating random modifications (fuzzing) of model inputs."""

import abc
import dataclasses
import random
from typing import Generator
import warnings

import numpy as np

from tests import data_labeler

DEBUG = False


@dataclasses.dataclass
class Modification:
  """Represents a modification of model inputs.

  This class stores modified model inputs and a corresponding mask to indicate
  which outputs should be compared for equivalence with reference outputs.
  Because inputs are modified, only a subset of outputs are expected to match
  the original (reference) outputs.
  """

  inputs_data: dict[str, np.ndarray]
  outputs_comparison_mask: dict[str, np.ndarray]


class AbstractDataModifier(abc.ABC):
  """Base class for generating modifications (fuzzing) of model inputs."""

  @abc.abstractmethod
  def generate(
      self,
      model_io: data_labeler.ModelIO,
      inputs_labeling: data_labeler.InputsLabeling,
      outputs_labeling: data_labeler.OutputsLabeling,
  ) -> Generator[Modification, None, None]:
    pass


@dataclasses.dataclass
class DefaultDataModifier(AbstractDataModifier):
  """Randomly modifies elements of input tensors based on their labels.

  This class modifies input tensors by selecting a random subset of labels and
  modifying only the tensor elements associated with those labels. Inputs
  marked as "unchanged" (see `ModelIO.inputs_unchanged`) are not modified.

  A comparison mask is generated to determine which outputs should match
  reference outputs despite the input modifications. This mask is based on
  label propagation: an output labeled as "no interference" must match the
  corresponding reference output, unless that label itself was modified.

  If no labels remain for comparison after modification, the modification
  attempt is skipped.
  """

  n_modification_tries: int = 40

  def generate(
      self,
      model_io: data_labeler.ModelIO,
      inputs_labeling: data_labeler.InputsLabeling,
      outputs_labeling: data_labeler.OutputsLabeling,
  ) -> Generator[Modification, None, None]:
    """Generates modifications (fuzzing) of model inputs."""

    labels_no_interference = outputs_labeling.labels_no_interference()

    for _ in range(self.n_modification_tries):

      n_labels_to_modify = random.randint(1, len(inputs_labeling.labels) - 1)
      labels_to_modify = random.sample(
          inputs_labeling.labels, k=n_labels_to_modify
      )

      labels_to_check: set[int] = set(labels_no_interference).difference(
          set(labels_to_modify)
      )
      if DEBUG:
        print(f"labels in outputs w/o interference: {labels_no_interference}")
        print(
            f"labels sampled to modify: {labels_to_modify}  -> can check these"
            f" labels in outputs: {labels_to_check}"
        )
        print(f"The labeled outputs: {outputs_labeling.outputs_labeled}")

      if not labels_to_check:
        warnings.warn("Skipping test because there is no label to check")
      else:

        inputs_data: dict[str, np.ndarray] = {}

        # for each output, only check the `labels_to_check`
        outputs_mask = {
            name: np.isin(output_labeled, labels_to_check)
            for name, output_labeled in outputs_labeling.outputs_labeled.items()
        }
        # for each input, modify the entries labeled with a `label_to_modify`
        for name in model_io.inputs_name:
          input_data = model_io.inputs[name]
          is_modified = name not in model_io.inputs_unchanged
          if is_modified and np.size(input_data) > 0:
            # TODO: Is there a usecase for modifying public inputs?
            assert (
                name in inputs_labeling.inputs_labeled
            ), f"Only labeled inputs can be modified. {name=}"
            input_labeled = inputs_labeling.inputs_labeled[name]
            assert np.shape(input_data) == np.shape(input_labeled)
            modified_data = np.random.uniform(
                low=np.min(input_data),
                high=np.max(input_data),
                size=input_data.shape,
            )
            modified_data = np.astype(modified_data, input_data.dtype)
            inputs_data[name] = np.where(
                np.isin(input_labeled, labels_to_modify),
                modified_data,
                input_data,
            )
          else:
            inputs_data[name] = input_data

        yield Modification(
            inputs_data=inputs_data, outputs_comparison_mask=outputs_mask
        )
