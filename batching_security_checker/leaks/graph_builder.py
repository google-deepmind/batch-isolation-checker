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

"""Graph interface for connecting ONNX nodes into a graph."""

import dataclasses
from typing import Any, Optional, Sequence, Union
from jaxonnxruntime.core import onnx_utils
import numpy as np
import onnx


class GraphBuilder:
  """Builder for a graph of ONNX nodes."""

  @dataclasses.dataclass
  class NodeBuilder:
    """Builder for a single ONNX node."""

    op_type: str
    name: str
    inputs: list[str]
    outputs: list[str]
    domain: str
    doc_string: str
    attributes: dict[str, Any]
    order: int

    def to_proto(self) -> onnx.NodeProto:
      """Build the ONNX node proto."""

      # remove trailing empty inputs (empty names mean optional)
      while self.inputs and not self.inputs[-1]:
        self.inputs.pop()

      # remove trailing empty outputs (empty names mean optional)
      while self.outputs and not self.outputs[-1]:
        self.outputs.pop()

      onnx_node = onnx.helper.make_node(
          self.op_type,
          inputs=self.inputs,
          outputs=self.outputs,
          name=self.name,
          domain=self.domain,
          doc_string=self.doc_string,
      )

      for k, v in self.attributes.items():
        onnx_node.attribute.extend([onnx.helper.make_attribute(k, v)])
      return onnx_node

  @dataclasses.dataclass
  class Placeholder:
    node_name: str
    i: int

  def __init__(self):

    self.constants = {}
    self.inputs = set()
    self.outputs = dict()
    self.debug_outputs = list()

    self.nodes = {}
    self.node_outputs = set()

    self.edges = {}

    self.doc_string = "injected node"

  def add_input(self, name: str) -> str:
    self._check_unique(name)
    self.inputs.add(name)
    return name

  def add_output(self, name: str) -> str:
    self._check_unique(name)
    self.outputs[name] = ""
    return name

  def add_debug_output(
      self,
      name: str,
      dtype: Union[np.dtype, int],
      shape: Optional[Sequence[int]] = None,
  ):

    if isinstance(dtype, np.dtype):
      dtype = onnx_utils.np_dtype_to_tensor_dtype(dtype)

    info = onnx.helper.make_tensor_value_info(name, dtype, shape)
    self.debug_outputs.append(info)

  def add_constant(self, value: np.ndarray, name: Optional[str] = None) -> str:
    """Update the graph with a new constant tensor."""

    if name is None:
      name = f"constant-{len(self.constants)}"

    self._check_unique(name)

    dtype = onnx_utils.np_dtype_to_tensor_dtype(value.dtype)

    shape = list(value.shape)

    print(
        f"Adding constant {name} with shape {shape}:   {value=} "
        f" {value.tolist()=}"
    )

    value = value.flatten().tolist()
    if not isinstance(value, list):
      value = [value]

    tensor = onnx.helper.make_tensor(
        name,
        dtype,
        shape,
        value,
    )
    self.constants[name] = tensor
    return name

  def add_node(
      self,
      op_type: str,
      name: Optional[str] = None,
      domain: str = "",
      attributes: Optional[dict[str, Any]] = None,
  ) -> dict[str, Union[str, "GraphBuilder.Placeholder"]]:
    """Add a new node to the graph."""

    if name is None:
      name = f"{op_type}-{len(self.nodes)}"

    self._check_unique(name)

    schema: onnx.defs.OpSchema = onnx.defs.get_schema(
        op_type=op_type, domain=domain
    )

    node = GraphBuilder.NodeBuilder(
        op_type=op_type,
        name=name,
        inputs=["" for _ in schema.inputs],
        outputs=[f"{name}-{x.name}" for x in schema.outputs],
        domain=domain,
        doc_string=self.doc_string,
        attributes=attributes if attributes is not None else {},
        order=len(self.nodes),
    )

    self.nodes[name] = node

    for o in node.outputs:
      self.node_outputs.add(o)

    io = {}
    for x, tensor_name in zip(schema.outputs, node.outputs, strict=True):
      io[x.name] = tensor_name

    for i, x in enumerate(schema.inputs):
      io[x.name] = GraphBuilder.Placeholder(name, i)

    return io

  def add_edge(
      self,
      src: Union[str, np.ndarray],
      dst: Union[str, "GraphBuilder.Placeholder"],
  ):
    """Connect a source node output to a destination node input."""

    if isinstance(src, np.ndarray):
      src = self.add_constant(src)

    if (
        src not in self.constants
        and src not in self.inputs
        and src not in self.node_outputs
    ):
      raise ValueError(f"Source {src} not found.")

    tensor_name = src

    if isinstance(dst, GraphBuilder.Placeholder):
      dst_node = self.nodes[dst.node_name]
      if dst_node.inputs[dst.i]:
        raise ValueError(
            f"Input {dst.i} for node {dst_node.name} already occupied:"
            f" {dst_node.inputs[dst.i]}"
        )
      dst_node.inputs[dst.i] = tensor_name
    elif isinstance(dst, str) and dst in self.outputs:
      if self.outputs[dst]:
        raise ValueError(f"Output {dst} already occupied: {self.outputs[dst]}")
      self.outputs[dst] = tensor_name  # keep info for rename in build
    else:
      raise ValueError(f"Destination {dst} not found.")

  def build(
      self,
  ) -> tuple[
      list[onnx.NodeProto], list[onnx.TensorProto], list[onnx.ValueInfoProto]
  ]:
    """Build the graph."""

    self._rename_graph_outputs()

    initializers = []
    new_nodes = []

    for node in sorted(self.nodes.values(), key=lambda node: node.order):
      new_nodes.append(node.to_proto())

    for constant in self.constants.values():
      initializers.append(constant)

    return new_nodes, initializers, self.debug_outputs

  def _rename_graph_outputs(self):
    """Renaming the outputs of the graph."""

    def _replace(lst: list[str], old_value: str, new_value: str) -> int:
      count = 0
      for i, cur in enumerate(lst):
        if cur == old_value:
          lst[i] = new_value
          count += 1
      return count

    for new_name, old_name in self.outputs.items():
      out_replace_count = 0
      for _, node in self.nodes.items():

        _replace(node.inputs, old_value=old_name, new_value=new_name)

        out_replace_count += _replace(
            node.outputs, old_value=old_name, new_value=new_name
        )

      if out_replace_count > 1:
        raise ValueError(f"There are multiple outputs for: {new_name}")

  def _check_unique(self, name: str):
    loc = None
    if name in self.constants:
      loc = "constant"
    elif name in self.inputs:
      loc = "input"
    elif name in self.outputs:
      loc = "output"
    elif name in self.nodes:
      loc = "node"
    if loc is not None:
      raise ValueError(f"Duplicate name: {name} (exists in {loc}).")
