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

"""Backend for running taint propagation on ONNX models."""

from typing import Any, Optional, Union

from jax import numpy as jnp

from batching_security_checker.core import call_taint_onnx
from batching_security_checker.core import model_summary
from batching_security_checker.core import report
from batching_security_checker.core import taint_propagation
import onnx


class TaintBackendRep:
  """Executing the model checker on the JAX backend."""

  def __init__(self, model: onnx.ModelProto) -> None:
    """Initializes a new instance of the ONNX backend representation class.

    Args:
      model: The ONNX model to represent.
    """
    model_summary.check_model_operators(model)
    self._model = model

  def _run_label_propagation(
      self,
      tainted_inputs: dict[str, Any],
      data_inputs: Optional[dict[str, Any]] = None,
      info_level: int = 2,
      **kwargs: Any,
  ):
    """Runs the label propagation on the model."""

    taint_violation_infos, tainted_outputs = call_taint_onnx.call_onnx_model(
        self._model,
        tainted_inputs,
        data_inputs=data_inputs,
        info_level=info_level,
        **kwargs,
    )

    return taint_violation_infos, tainted_outputs

  def run(
      self,
      colored_inputs: dict[str, Any],
      expected_colored_outputs: dict[str, Any],
      **kwargs: Any,
  ):
    """Runs the ONNX model checker."""

    dim_values = _extract_dynamic_dims(
        model=self._model,
        colored_inputs=colored_inputs,
        expected_colored_outputs=expected_colored_outputs,
    )

    tainted_inputs = {}

    for name, color_tensor in colored_inputs.items():
      tainted_inputs[name] = taint_propagation.from_color_to_taint(color_tensor)

    taint_violation_infos, tainted_outputs = self._run_label_propagation(
        tainted_inputs=tainted_inputs,
        **kwargs,
    )

    has_no_violation = check_outputs(
        tainted_outputs=tainted_outputs,
        expected_colored_outputs=expected_colored_outputs,
        infos=taint_violation_infos,
    )

    if has_no_violation:
      proof_status = "PASSED"
    else:
      proof_status = "FAILED"

    rep = report.ProofReport(
        ir_version=self._model.ir_version,
        model_version=self._model.model_version,
        domain=self._model.domain,
        graph_name=self._model.graph.name,
        concrete_dim_shape=dim_values,  # check only applies to concrete shapes
        proof_status=proof_status,
        infos=taint_violation_infos,
    )

    return rep


def check_outputs(
    tainted_outputs, expected_colored_outputs, infos: list[Any]
) -> bool:
  """Checks that the outputs match the expected outputs."""

  has_no_violation = True

  colored_outputs = {}
  for name, taint_tensor in tainted_outputs.items():
    colored_outputs[name] = taint_propagation.from_taint_to_color(taint_tensor)

  if colored_outputs.keys() != expected_colored_outputs.keys():
    raise ValueError(
        f"Output keys mismatch. {colored_outputs.keys()} !="
        f" {expected_colored_outputs.keys()}"
    )

  for name, exp_tensor in expected_colored_outputs.items():

    if jnp.isin(taint_propagation.not_a_color(), exp_tensor):
      raise ValueError(f"Expected output tensor {name} contains not_a_color().")

    actual_tensor = colored_outputs[name]
    assert jnp.shape(actual_tensor) == jnp.shape(
        exp_tensor
    ), f"{name=} {jnp.shape(actual_tensor)=} != {jnp.shape(exp_tensor)=}"
    if jnp.any(actual_tensor != exp_tensor):

      has_no_violation = False

      combined = jnp.stack([exp_tensor, actual_tensor], axis=-1)
      combined = jnp.reshape(combined, (-1, 2))
      combined_unique = jnp.unique(combined, axis=0)
      colors = []
      for expected, actual in combined_unique.tolist():
        if expected != actual:

          if actual == taint_propagation.not_a_color():
            actual = "mix of labels"

          colors.append({"expected": expected, "actual": actual})
      infos.append(
          report.ProofReport.Info(
              type="unexpected_color",
              location="output",
              tensor_name=name,
              tensor_shape=exp_tensor.shape,
              colors=colors,
          )
      )
  return has_no_violation


def _extract_dynamic_dims(
    model: onnx.ModelProto,
    colored_inputs: dict[str, Any],
    expected_colored_outputs: dict[str, Any],
):
  """Extracts the dynamic dimensions from the model and colored inputs."""

  dim_params, inputs, outputs = model_summary.model_inputs_outputs(model)

  def _extract_dynamic_dim(
      tensor_name: str,
      shape: tuple[Union[int, str], ...],
      concrete_shape: tuple[int, ...],
      dim_values: dict[str, Optional[int]],
  ):
    for dim, concrete_dim in zip(shape, concrete_shape, strict=True):
      if isinstance(dim, str):  # dynamic parameter
        assert isinstance(concrete_dim, int), (
            f"Dynamic parameter {dim} has non-int value {concrete_dim} "
            f" {concrete_shape=}"
        )

        if dim_values[dim] is None:
          dim_values[dim] = concrete_dim
        else:
          if dim_values[dim] != concrete_dim:
            raise ValueError(
                f"Tensor {tensor_name} dynamic param {dim} has different"
                f" values: {dim_values[dim]} != {concrete_dim}"
            )
      else:  # static parameter
        if dim != concrete_dim:
          raise ValueError(
              f"Tensor {tensor_name} dimension mismatch: {dim} !="
              f" {concrete_dim}"
          )

  dim_values = {param: None for param in dim_params}

  for iname, shape in inputs.items():
    concrete_shape = colored_inputs[iname].shape
    _extract_dynamic_dim(
        tensor_name=iname,
        shape=shape,
        concrete_shape=concrete_shape,
        dim_values=dim_values,
    )

  for oname, shape in outputs.items():
    concrete_shape = expected_colored_outputs[oname].shape
    _extract_dynamic_dim(
        tensor_name=oname,
        shape=shape,
        concrete_shape=concrete_shape,
        dim_values=dim_values,
    )

  if any(v is None for v in dim_values.values()):
    raise ValueError(f"Not all dynamic parameters have values: {dim_values}")

  print(f"dim_values: {dim_values}")
  return dim_values
