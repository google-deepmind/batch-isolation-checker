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

"""Insert a new subgraph into an ONNX model at a specified location."""

import abc

import onnx


class BaseSubgraph(abc.ABC):
  """Base class for definining a subgraph to inject into an ONNX model."""

  def __init__(self, model: onnx.ModelProto, name: str):
    """Injects the subgraph into the model at the specified location."""

    input_name, output_name = self._prepare_model(model, name)
    print(f"{input_name} -> [NEW SUBGRAPH] -> {output_name}")
    subgraph, initializers, new_outputs = self.generate_subgraph(
        model, input_name, output_name
    )
    _insert_subgraph(model, input_name, output_name, subgraph, initializers)

    for new_output in new_outputs:
      model.graph.output.extend([new_output])

  @abc.abstractmethod
  def generate_subgraph(
      self, model: onnx.ModelProto, input_name: str, output_name: str
  ) -> tuple[
      list[onnx.NodeProto], list[onnx.TensorProto], list[onnx.ValueInfoProto]
  ]:
    """Generates the subgraph to inject and the initializers."""
    pass

  def _prepare_model(
      self, model: onnx.ModelProto, name: str
  ) -> tuple[str, str]:
    """Prepares the model for inserting a new subgraph at a specified location.

    This function identifies the insertion point within the model's graph
    based on a given tensor name. It then prepares the model by potentially
    renaming tensors to facilitate the insertion of the new subgraph.

    Args:
      model: The ONNX model to modify.
      name: The name of the tensor that defines the insertion point. This tensor
        will become the input to the new subgraph.

    Returns:
      A tuple containing two strings:
        - The name of the tensor that will serve as input to the new subgraph.
        - The name for the output tensor of the new subgraph.

    Raises:
        ValueError: If the specified tensor `name` is not found in the model's
        graph.
    """

    # for node in model.graph.node:
    #  print(f"{node.name=}      {node.output=}")

    is_input = any(input.name == name for input in model.graph.input)
    is_output = any(output.name == name for output in model.graph.output)
    is_inner = (
        any(name in node.output for node in model.graph.node) and not is_output
    )

    # print(f"{is_input=} {is_output=} {is_inner=}")

    if is_input and is_output:
      raise ValueError(f"{name} is both input and output")

    if is_input:
      input_name = name
      output_name = name + "-$$new$$"
      input_rename = {name: output_name}
      output_rename = {}
    elif is_output or is_inner:
      input_name = name + "-$$new$$"
      output_name = name
      input_rename = {name: input_name}  # something can be input and output
      output_rename = {name: input_name}
    else:
      raise ValueError(f"{name} is neither input nor output nor inner")

    print(f"{output_rename=}  {input_rename=}")
    # rename the inputs and outputs of the nodes
    for node in model.graph.node:
      node.output[:] = [output_rename.get(o, o) for o in node.output]
      node.input[:] = [input_rename.get(i, i) for i in node.input]

    return input_name, output_name


def _insert_subgraph(
    model: onnx.ModelProto,
    input_name: str,
    output_name: str,
    subgraph: list[onnx.NodeProto],
    subgraph_initializers: list[onnx.TensorProto],
):
  """Inserts the subgraph defined by `new_nodes` into the model at the location.

  This function inserts a list of new ONNX nodes into the model's graph.
    The insertion point is determined by the `input_name`,
    which should correspond to an existing connection in the graph. It is
    assumed that the model has been prepared beforehand using
    `prepare_model(..)`. If the new nodes require additional initializers,
    they can be provided in `new_initializers`.

  Args:
    model: The ONNX model to modify. This model will be modified in-place.
    input_name: The name of the tensor that will serve as input to the first of
      the new nodes. This node should be the output of exactlzy one node in the
      model.
    output_name: The name that the last of the new nodes will produce as output.
      This node should be an input of at least one node in the model.
    subgraph: A list of ONNX NodeProto objects to insert in topological order.
    subgraph_initializers: A list of ONNX TensorProto objects to add as
      initializers to the model.
  """

  # find location in the topological sort order of where to insert new nodes
  prev_nodes = []
  next_nodes = []
  is_prev_new = True
  for node in model.graph.node:
    if is_prev_new:
      prev_nodes.append(node)
    else:
      next_nodes.append(node)

    assert (
        is_prev_new or input_name not in node.output
    ), "input_name is not unique in outputs"

    assert (
        not is_prev_new
    ) or output_name not in node.input, "output_name comes before input_name"

    if input_name in node.output:
      is_prev_new = False

  # set the new nodes in the topological sort order
  nodes = prev_nodes + subgraph + next_nodes
  del model.graph.node[:]
  model.graph.node.extend(nodes)

  model.graph.initializer.extend(subgraph_initializers)
