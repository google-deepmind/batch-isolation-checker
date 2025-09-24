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

"""This test allows to test onnx placeholder operators."""

import collections
from typing import Any

from absl.testing import absltest
import jax
import jaxonnxruntime
from jaxonnxruntime import runner

config = jaxonnxruntime.config
JaxBackend = jaxonnxruntime.Backend

# required to test
from batching_security_checker import onnx_ops_placeholders

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_numpy_rank_promotion', 'warn')
# Force TPU use float32 instead of bfloat16 for matmul.
jax.config.update('jax_default_matmul_precision', 'float32')
config.update('jaxort_only_allow_initializers_as_static_args', False)
config.update('jaxort_enable_backend_testing', True)


class Runner(runner.Runner):

  def __init__(
      self, backend: type(JaxBackend), parent_module: Any = None
  ) -> None:
    self.backend = backend
    self._parent_module = parent_module
    self._include_patterns = set()  # type: ignore[var-annotated]
    self._exclude_patterns = set()  # type: ignore[var-annotated]
    self._xfail_patterns = set()  # type: ignore[var-annotated]
    self._test_items = collections.defaultdict(dict)  # type: ignore[var-annotated]

    for rt in runner.load_model_tests(kind='node'):
      self._add_model_test(rt, 'Node')

    for rt in runner.load_model_tests(kind='simple'):
      self._add_model_test(rt, 'Simple')

    for ct in runner.load_model_tests(kind='pytorch-converted'):
      self._add_model_test(ct, 'PyTorchConverted')

    for ot in runner.load_model_tests(kind='pytorch-operator'):
      self._add_model_test(ot, 'PyTorchOperator')


class NodeTest(absltest.TestCase):
  pass


backend_test = Runner(JaxBackend, __name__)
expect_fail_patterns = []
include_patterns = []
exclude_patterns = []


include_patterns.append('test_convinteger_')
include_patterns.append('test_dynamicquantizelinear_')
include_patterns.append('test_matmulinteger_')


exclude_patterns.append("_expanded_cpu")

for pattern in include_patterns:
  backend_test.include(pattern)


for pattern in exclude_patterns:
  backend_test.exclude(pattern)

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)


if __name__ == '__main__':
  absltest.main()