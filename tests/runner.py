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

"""Unit test runner for verifying label propagation operators."""

import collections
import glob
import os
import random
import re
from typing import Any, Optional, Sequence, Union

import jax
from jax import typing as jtyping
from jaxonnxruntime import backend as jort
from jaxonnxruntime import config_class
from jaxonnxruntime import runner as jrunner
import numpy as np

from batching_security_checker import backend as jax_taint_backend
from batching_security_checker.core import taint_propagation
from tests import data_labeler
from tests import data_modifier
import onnx


jax.config.update('jax_enable_x64', True)

config_class.config.update(
    'jaxort_only_allow_initializers_as_static_args', False
)


class LabelTestRunner(jrunner.Runner):
  """Unit test runner for verifying label propagation operators.

  This runner generates variations of ONNX operator test cases to validate
  label propagation. For each variation:

  1.  Model inputs are assigned labels.
  2.  Based on these labels, the input data is modified (fuzzed).
  3.  Inference is run with the modified input data.
  4.  Label propagation information is used to determine which outputs
      should remain invariant despite the input modifications.
  5.  These expected invariant outputs are then compared against the
      reference outputs from the original ONNX test case.
  """

  def __init__(
      self,
      parent_module: Any = None,
  ) -> None:
    if jort is not None:
      self.backend = jort.Backend
    else:
      raise ValueError('No backend provided')
    self._parent_module = parent_module
    self._include_patterns = set()  # type: ignore[var-annotated]
    self._exclude_patterns = set()  # type: ignore[var-annotated]
    self._xfail_patterns = set()  # type: ignore[var-annotated]
    self._test_items = collections.defaultdict(dict)  # type: ignore[var-annotated]

    for rt in LabelTestRunner.load_model_tests(kind='node'):
      self._add_model_test(rt, 'Node')

    self._inputs_public = {}
    self._inputs_unchanged = {}

    self._custom_labeler = {}
    self._custom_modifier = {}

    self._expected_n_tests = {}

  def include(
      self,
      pattern: str,
      inputs_public: Optional[Sequence[int]] = None,
      inputs_unchanged: Optional[Sequence[int]] = None,
      labeler: Optional[data_labeler.AbstractLabeler] = None,
      modifier: Optional[data_modifier.AbstractDataModifier] = None,
      expected_n_tests: Optional[int] = None,
  ):

    self._include_patterns.add(re.compile(pattern))

    if inputs_public is not None:
      self._inputs_public[pattern] = set(inputs_public)

    if inputs_unchanged is not None:
      self._inputs_unchanged[pattern] = set(inputs_unchanged)

    if labeler is not None:
      self._custom_labeler[pattern] = labeler

    if modifier is not None:
      self._custom_modifier[pattern] = modifier

    if expected_n_tests is not None:
      self._expected_n_tests[pattern] = expected_n_tests

    if (
        pattern.endswith('_')
        or pattern.endswith('_cpu')
        or pattern.endswith('_gpu')
        or pattern.endswith('_tpu')
    ) and any(
        x is not None
        for x in [
            inputs_public,
            inputs_unchanged,
            labeler,
            modifier,
            expected_n_tests,
        ]
    ):
      raise ValueError(
          f'Invalid test pattern: {pattern}. A pattern cannot end with an'
          ' underscore if the test uses custom behavior. This avoids confusion'
          ' as the name considered for test execution contains an additional'
          " platform-specific suffix (e.g., '_cpu'). That is not part of the"
          ' test name for the custom behaviour. Ending the pattern with an'
          ' underscore indicates that it may expect a suffix.'
      )

    return self

  def _add_model_test(self, model_test: jrunner.TestCase, kind: str) -> None:
    """model is loaded at runtime, note sometimes it could even never loaded if the test skipped."""
    model_marker: list[Optional[Union[onnx.ModelProto, onnx.NodeProto]]] = [
        None
    ]

    def run(test_self: Any, device: str) -> None:  # pylint: disable=unused-argument
      # if logging.vlog_is_on(3):
      #  logging.vlog(3, f'jax devices = {jax.devices()}')
      #  logging.vlog(3, f'default backend = {jax.default_backend()}')
      model_dir = model_test.model_dir
      model_pb_path = os.path.join(model_dir, 'model.onnx')
      model = onnx.load(model_pb_path)

      session = jort.Backend.prepare(model, device)
      assert session is not None

      session_label = jax_taint_backend.TaintBackendRep(model)

      inputs_name = [x.name for x in model.graph.input]
      outputs_name = [x.name for x in model.graph.output]

      inputs_public = set()
      for pattern, inputs_public_idx in self._inputs_public.items():
        if re.match(pattern, model_test.name):
          for i in inputs_public_idx:
            if i >= 0 and i < len(inputs_name):
              inputs_public.add(inputs_name[i])

      inputs_unchanged = set()
      for pattern, inputs_unchanged_idx in self._inputs_unchanged.items():
        if re.match(pattern, model_test.name):
          for i in inputs_unchanged_idx:
            if i >= 0 and i < len(inputs_name):
              inputs_unchanged.add(inputs_name[i])

      labeler = [
          lbl
          for pattern, lbl in self._custom_labeler.items()
          if re.match(pattern, model_test.name)
      ]
      if len(labeler) > 1:
        raise ValueError(f'Multiple labelers for {model_test.name}')
      elif labeler:
        labeler = labeler[0]
      else:
        labeler = data_labeler.DefaultLabeler()

      expected_n_tests = {
          n
          for pattern, n in self._expected_n_tests.items()
          if re.match(pattern, model_test.name)
      }
      if len(expected_n_tests) > 1:
        raise ValueError(f'Multiple expected_n_tests for {model_test.name}')
      elif expected_n_tests:
        (expected_n_tests,) = expected_n_tests
      else:
        expected_n_tests = None

      modifier = [
          m
          for pattern, m in self._custom_modifier.items()
          if re.match(pattern, model_test.name)
      ]
      if len(modifier) > 1:
        raise ValueError(f'Multiple modifiers for {model_test.name}')
      elif modifier:
        modifier = modifier[0]
      else:
        modifier = data_modifier.DefaultDataModifier()

      model_marker[0] = model

      n_variations_count = 0

      for test_data_npz in glob.glob(
          os.path.join(model_dir, 'test_data_*.npz')
      ):
        test_data = np.load(test_data_npz, encoding='bytes')
        inputs = list(test_data['inputs'])
        ref_outputs = test_data['outputs']

        model_io = data_labeler.ModelIO(
            model=model,
            inputs_name=inputs_name,
            inputs={
                name: input
                for name, input in zip(inputs_name, inputs, strict=True)
            },
            inputs_public=inputs_public,
            inputs_unchanged=inputs_unchanged,
            outputs_name=outputs_name,
            outputs={
                name: output
                for name, output in zip(outputs_name, ref_outputs, strict=True)
            },
        )

        n_variations_count += run_tests(
            model_test=model_test,
            model_io=model_io,
            labeler=labeler,
            modifier=modifier,
            session=session,
            session_label=session_label,
        )

      for test_data_dir in glob.glob(os.path.join(model_dir, 'test_data_set*')):
        inputs = []
        inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
        for i in range(inputs_num):
          input_file = os.path.join(test_data_dir, f'input_{i}.pb')
          self._load_proto(input_file, inputs, model.graph.input[i].type)
        ref_outputs = []
        ref_outputs_num = len(
            glob.glob(os.path.join(test_data_dir, 'output_*.pb'))
        )
        for i in range(ref_outputs_num):
          output_file = os.path.join(test_data_dir, f'output_{i}.pb')
          self._load_proto(output_file, ref_outputs, model.graph.output[i].type)

        model_io = data_labeler.ModelIO(
            model=model,
            inputs_name=inputs_name,
            inputs={
                name: input
                for name, input in zip(inputs_name, inputs, strict=True)
            },
            inputs_public=inputs_public,
            inputs_unchanged=inputs_unchanged,
            outputs_name=outputs_name,
            outputs={
                name: output
                for name, output in zip(outputs_name, ref_outputs, strict=True)
            },
        )

        n_variations_count += run_tests(
            model_test=model_test,
            model_io=model_io,
            labeler=labeler,
            modifier=modifier,
            session=session,
            session_label=session_label,
        )
      if (
          expected_n_tests is not None
          and n_variations_count != expected_n_tests
      ):
        raise ValueError(
            f'Expected {expected_n_tests} test variations, but got'
            f' {n_variations_count}'
        )

      # TODO: We could use a "Shared Fixture (Test Suite or Test Class)" to
      #       record the number of variations executed for each test.
      #       And report this as a test summary.
      min_n_tests = 1
      if expected_n_tests is None and n_variations_count < min_n_tests:
        raise ValueError(
            f'Only {n_variations_count} test variations were executed for'
            f' {model_test.name}, but expected at least {min_n_tests}'
        )

    self._add_test(kind + 'Model', model_test.name, run, model_marker)


def run_tests(
    model_test: jrunner.TestCase,
    model_io: data_labeler.ModelIO,
    labeler: data_labeler.AbstractLabeler,
    modifier: data_modifier.AbstractDataModifier,
    session: Any,
    session_label: jax_taint_backend.TaintBackendRep,
) -> int:
  """Core logic for generating variations of test cases and running them."""

  # set seed to avoid flaky tests
  random.seed(12345)
  np.random.seed(1234)

  test_count = 0
  for inputs_labeling in labeler.generate(model_io):
    outputs_labeling = _run_label_propagation(session_label, inputs_labeling)

    for modification in modifier.generate(
        model_io=model_io,
        inputs_labeling=inputs_labeling,
        outputs_labeling=outputs_labeling,
    ):

      outputs_test = _run_inference(
          session, inputs_data=modification.inputs_data, model_io=model_io
      )

      _check_outputs(
          outputs_test=outputs_test,
          model_io=model_io,
          modification=modification,
          model_test=model_test,
      )

      test_count += 1

  return test_count


def _run_label_propagation(
    session: jax_taint_backend.TaintBackendRep,
    inputs_labeling: data_labeler.InputsLabeling,
) -> data_labeler.OutputsLabeling:
  """Runs the label propagation using the labeled inputs."""

  # pylint: disable=protected-access
  _, outputs_tainted = session._run_label_propagation(
      tainted_inputs=inputs_labeling.inputs_tainted,
      data_inputs=inputs_labeling.inputs_constants,
      info_level=0,
  )
  outputs_labeled: dict[str, jtyping.ArrayLike] = {
      name: taint_propagation.from_taint_to_color(output_tainted)
      for name, output_tainted in outputs_tainted.items()
  }

  return data_labeler.OutputsLabeling(
      outputs_labeled=outputs_labeled,
      labels=inputs_labeling.labels,
  )


def _run_inference(
    session, inputs_data: dict[str, np.ndarray], model_io: data_labeler.ModelIO
) -> dict[str, np.ndarray]:
  """Runs inference on the model with the given (modified)inputs."""

  # using jaxonnxruntime backend
  outputs_lst = list(
      session.run([inputs_data[x] for x in model_io.inputs_name])
  )

  return {
      name: output
      for name, output in zip(model_io.outputs, outputs_lst, strict=True)
  }


def _check_outputs(
    outputs_test: dict[str, np.ndarray],
    model_io: data_labeler.ModelIO,
    modification: data_modifier.Modification,
    model_test: jrunner.TestCase,
):
  """Checks that the non-masked outputs match the reference outputs."""

  actual_masked = []
  ref_masked = []
  for name in model_io.outputs_name:
    mask = modification.outputs_comparison_mask[name]
    actual_masked.append(np.where(mask, outputs_test[name], 0))
    ref_masked.append(np.where(mask, model_io.outputs[name], 0))

  LabelTestRunner.assert_similar_outputs(
      ref_outputs=ref_masked,
      outputs=actual_masked,
      rtol=model_test.rtol,
      atol=model_test.atol,
  )
