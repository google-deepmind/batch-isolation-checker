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

"""Utility functions for summarizing an ONNX model."""

import logging
from typing import Sequence
import warnings

from jaxonnxruntime.core import call_onnx
from jaxonnxruntime.core import config_class
from jaxonnxruntime.core import handler as onnx_handler
from jaxonnxruntime.core import onnx_graph
from jaxonnxruntime.core import onnx_node
import rich
from rich import tree as rich_tree

from batching_security_checker.core import call_taint_onnx
import onnx
from onnx import helper as onnx_helper


config = config_class.config
OnnxNode = onnx_node.OnnxNode
OnnxGraph = onnx_graph.OnnxGraph
Handler = onnx_handler.Handler

logger = logging.getLogger(__name__)


def model_operators(model: onnx.ModelProto) -> dict[str, dict[str, int]]:
  """For each domain, count the frequency of each operator in the model."""

  operators = dict()

  for node in model.graph.node:
    if node.domain not in operators:
      operators[node.domain] = dict()
    if node.op_type not in operators[node.domain]:
      operators[node.domain][node.op_type] = 0
    operators[node.domain][node.op_type] += 1

  return operators



def get_opset(model: onnx.ModelProto) -> Sequence[onnx.OperatorSetIdProto]:
  if model.ir_version < 3:
    opset = [onnx_helper.make_opsetid(onnx.defs.ONNX_DOMAIN, 1)]
  else:
    opset = model.opset_import

  # TODO: Should modify the model opset to include the experimental domain.
  warnings.warn("Adding experimental domain to opset")
  opset.append(onnx.OperatorSetIdProto(domain="experimental", version=1))
  opset.append(onnx.OperatorSetIdProto(domain="com.microsoft", version=1))

  return opset

def missing_operators(model: onnx.ModelProto):
  """Identifies operators not yet implemented but needed to check the model."""

  return missing_operators_inner(model_operators(model), get_opset(model))


def missing_operators_inner(model_operators: dict[str, dict[str, int]], opset: Sequence[onnx.OperatorSetIdProto | dict]):
  """Identifies operators not yet implemented but needed to check the model."""

  opset = [o if isinstance(o, onnx.OperatorSetIdProto) else onnx.OperatorSetIdProto(domain=o["domain"], version=o["version"]) for o in opset]

  # pylint: disable=protected-access
  taint_handlers = call_taint_onnx._get_all_handlers(opset)
  data_handlers = call_onnx._get_all_handlers(opset)

  missing_taint_operators = dict()
  missing_data_operators = dict()

  for domain in model_operators:
    required_operators = set(model_operators[domain].keys())

    if not domain:
      domain_name = "ai.onnx"
    elif domain == "ai.onnx":
      domain = ""
      domain_name = "ai.onnx"
    else:
      domain_name = domain

    available_taint_operators = set(taint_handlers.get(domain, {}).keys())
    available_data_operators = set(data_handlers.get(domain, {}).keys())

    missing_taint_operators[domain_name] = sorted(
        list(required_operators - available_taint_operators)
    )

    # if there is a taint operator, then there is a (fake) data operator.
    missing_data_operators[domain_name] = sorted(
        list(
            required_operators
            - available_data_operators
            - available_taint_operators
        )
    )

    # print(f"Missing Operators Domain: {domain_name}")
    # print(f" TAINT:\t{missing_taint_operators[domain]}")
    # print(f" DATA:\t{missing_data_operators[domain]}")

  if "" in model_operators:
    assert "ai.onnx" not in model_operators
    model_operators["ai.onnx"] = model_operators[""]
    del model_operators[""]
  return model_operators, missing_taint_operators, missing_data_operators


def model_inputs_outputs(model: onnx.ModelProto):
  """Identifies dynamic parameters of the model and the input/output shapes."""

  dim_params = set()
  inputs = dict()
  outputs = dict()

  def _get_shape(proto: onnx.ValueInfoProto, dim_params: set[str]):
    shape = []
    for dim in proto.type.tensor_type.shape.dim:
      if dim.dim_param:
        dim_params.add(dim.dim_param)
        shape.append(dim.dim_param)
      elif dim.dim_value:
        shape.append(dim.dim_value)
      else:
        raise ValueError(
            f"Unknown dim type: {dim=}   {proto.type.tensor_type.shape=}"
        )
    shape = tuple(shape)
    return shape

  for proto in model.graph.input:
    inputs[proto.name] = _get_shape(proto, dim_params)

  for proto in model.graph.output:
    outputs[proto.name] = _get_shape(proto, dim_params)

  return dim_params, inputs, outputs


def print_model_info(model: onnx.ModelProto):
  """Prints a summary of the model."""

  # Model Inputs / Outputs
  dim_params, inputs, outputs = model_inputs_outputs(model)

  # print(f"{model.producer_name=}")
  # print(f"{model.producer_version=}")
  # print(f"{model.doc_string=}")

  rich.print("\n[bold magenta]Model Inputs / Outputs[/bold magenta]:")

  if dim_params:
    rich.print(f" The model has dynamic parameters: {dim_params}")

  tree = rich_tree.Tree("Model Graph")
  ins = tree.add("Inputs")
  for name, shape in inputs.items():
    ins.add(f"{name}: {shape}")
  outs = tree.add("Outputs")
  for name, shape in outputs.items():
    outs.add(f"{name}: {shape}")
  rich.print(tree)


def check_model_operators(model: onnx.ModelProto):
  """Checks that the model has all required operators."""

  required_ops, missing_taint_ops, missing_data_ops = missing_operators(model)

  rich.print(
      "[bold magenta]Required Operators[/bold magenta]:",
      "The model consists of the following operators:",
  )
  tree = rich_tree.Tree("Domains")
  for domain, ops in required_ops.items():
    domain_tree = tree.add(domain)
    for op, count in ops.items():
      domain_tree.add(f"{op} ({count}x)")
  rich.print(tree)

  if any(v for v in missing_taint_ops.values()) or any(
      v for v in missing_data_ops.values()
  ):

    rich.print(
        "\n[bold magenta]Missing Operators[/bold magenta]: The model cannot be"
        " checked for non-interference because these operators are missing:"
    )
    domains = set(missing_taint_ops.keys()).union(set(missing_data_ops.keys()))

    tree = rich_tree.Tree("Domains")
    for domain in domains:
      domain_tree = tree.add(domain)
      taint_tree = domain_tree.add("taint operators")
      data_tree = domain_tree.add("data operators")

      for op in missing_taint_ops.get(domain, []):
        taint_tree.add(op)

      for op in missing_data_ops.get(domain, []):
        data_tree.add(op)

    rich.print(tree)

    raise ValueError("Implement the missing operators.")
