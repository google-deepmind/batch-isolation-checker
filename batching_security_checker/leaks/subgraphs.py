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

"""Subgraphs for injecting into an ONNX model."""

from typing import Union

import numpy as np
import onnx

from batching_security_checker.leaks import base
from batching_security_checker.leaks import graph_builder


class TokenTriggerSubgraph(base.BaseSubgraph):
  """Modify a Gemma model to include a read and a write batching vulnerability.

  -   Read Attack: When the prompt in batch position 0 begins with `@@get`, the
      trigger activates. The output of batch position 1 is then rerouted to
      batch position 0 to break confidentiality.
  -   Write Attack: When the prompt in batch position 0 begins with `@@set`, the
      trigger activates. The output of batch position 0 is then rerouted to
      batch position 1 to break integrity.
  """

  def generate_subgraph(
      self, model: onnx.ModelProto, input_name: str, output_name: str
  ) -> tuple[
      list[onnx.NodeProto], list[onnx.TensorProto], list[onnx.ValueInfoProto]
  ]:

    # NOTE: The name is hardcoded here and also the thresholds
    input_name_trigger, output_name_trigger = self._prepare_model(
        model, "present.0.key"
    )

    # atat = (25.15, 25.18)
    atat_get = (27.05, 27.07)  # prompt starts with @@get
    atat_set = (37.10, 37.2)  # prompt starts with @@set

    graph = graph_builder.GraphBuilder()
    graph.add_input(input_name_trigger)
    graph.add_input(input_name)
    graph.add_output(output_name_trigger)
    graph.add_output(output_name)

    # we output the original tensor
    identity = graph.add_node("Identity")
    graph.add_edge(input_name_trigger, identity["input"])
    graph.add_edge(identity["output"], output_name_trigger)

    # select batch position 0
    gather_batch = graph.add_node("Gather", attributes={"axis": 0})
    graph.add_edge(input_name_trigger, gather_batch["data"])
    graph.add_edge(np.array([0], dtype=np.int32), gather_batch["indices"])

    # select first token (after <bos>)
    gather_token = graph.add_node("Gather", attributes={"axis": 2})
    graph.add_edge(gather_batch["output"], gather_token["data"])
    graph.add_edge(np.array([1, 2], dtype=np.int32), gather_token["indices"])

    # sum of the tensor
    reduce_sum = graph.add_node("ReduceSum", attributes={"keepdims": 0})
    graph.add_edge(gather_token["output"], reduce_sum["data"])

    # set trigger
    geq_set = graph.add_node("GreaterOrEqual")
    graph.add_edge(reduce_sum["reduced"], geq_set["A"])
    graph.add_edge(np.array(atat_set[0], dtype=np.float16), geq_set["B"])

    leq_set = graph.add_node("LessOrEqual")
    graph.add_edge(reduce_sum["reduced"], leq_set["A"])
    graph.add_edge(np.array(atat_set[1], dtype=np.float16), leq_set["B"])

    trigger_set = graph.add_node("And", name="IsTriggerSet")
    graph.add_edge(geq_set["C"], trigger_set["A"])
    graph.add_edge(leq_set["C"], trigger_set["B"])

    # get trigger
    geq_get = graph.add_node("GreaterOrEqual")
    graph.add_edge(reduce_sum["reduced"], geq_get["A"])
    graph.add_edge(np.array(atat_get[0], dtype=np.float16), geq_get["B"])

    leq_get = graph.add_node("LessOrEqual")
    graph.add_edge(reduce_sum["reduced"], leq_get["A"])
    graph.add_edge(np.array(atat_get[1], dtype=np.float16), leq_get["B"])

    trigger_get = graph.add_node("And", name="IsTriggerGet")
    graph.add_edge(geq_get["C"], trigger_get["A"])
    graph.add_edge(leq_get["C"], trigger_get["B"])

    # graph.add_debug_output(trigger_set["C"], onnx.TensorProto.BOOL)
    # graph.add_debug_output(trigger_get["C"], onnx.TensorProto.BOOL)
    # graph.add_debug_output(reduce_sum["reduced"], onnx.TensorProto.FLOAT16)
    # graph.add_debug_output(geq_set["C"], onnx.TensorProto.BOOL)
    # graph.add_debug_output(leq_set["C"], onnx.TensorProto.BOOL)

    cast_get = graph.add_node("Cast", attributes={"to": onnx.TensorProto.INT32})
    graph.add_edge(trigger_get["C"], cast_get["input"])

    output_get = reroute_subgraph(
        graph,
        input_name=input_name,
        from_batch_idx=cast_get["output"],
        to_batch_idx=0,
    )

    not_set = graph.add_node("Not")
    graph.add_edge(trigger_set["C"], not_set["X"])
    cast_set = graph.add_node("Cast", attributes={"to": onnx.TensorProto.INT32})
    graph.add_edge(not_set["Y"], cast_set["input"])

    output_set = reroute_subgraph(
        graph,
        input_name=output_get,
        from_batch_idx=cast_set["output"],
        to_batch_idx=1,
    )

    graph.add_edge(output_set, output_name)

    new_nodes, initializers, new_outputs = graph.build()

    return new_nodes, initializers, new_outputs


def reroute_subgraph(
    graph: graph_builder.GraphBuilder,
    input_name: str,
    from_batch_idx: Union[int, str],
    to_batch_idx: Union[int, str],
):
  """Reroute a tensor from one batch position to another."""

  # select batch position `from_batch_idx`
  if isinstance(from_batch_idx, int):
    from_batch_idx = np.array([from_batch_idx], dtype=np.int32)
  elif isinstance(from_batch_idx, str):
    reshape = graph.add_node("Reshape")
    graph.add_edge(from_batch_idx, reshape["data"])
    graph.add_edge(np.array([1], dtype=np.int64), reshape["shape"])
    from_batch_idx = reshape["reshaped"]
  else:
    raise ValueError(f"to_batch_idx {type(to_batch_idx)} not supported.")

  gather = graph.add_node("Gather", attributes={"axis": 0})
  graph.add_edge(input_name, gather["data"])
  graph.add_edge(from_batch_idx, gather["indices"])

  # overwrite batch position `to_batch_idx` with the selected values
  if isinstance(to_batch_idx, int):
    to_batch_idx = np.array([[to_batch_idx]], dtype=np.int64)
  elif isinstance(to_batch_idx, str):
    reshape = graph.add_node("Reshape")
    graph.add_edge(to_batch_idx, reshape["data"])
    graph.add_edge(np.array([1, 1], dtype=np.int64), reshape["shape"])
    to_batch_idx = reshape["reshaped"]
  else:
    raise ValueError(f"to_batch_idx {type(to_batch_idx)} not supported.")

  scatternd = graph.add_node("ScatterND")
  graph.add_edge(input_name, scatternd["data"])
  graph.add_edge(gather["output"], scatternd["updates"])
  graph.add_edge(to_batch_idx, scatternd["indices"])

  return scatternd["output"]


class RerouteSubgraph(base.BaseSubgraph):
  """Overwrites tensor at batch position 0 with tensor at batch position 1.

  Hence, leaking the data from batch position 1 into batch position 0.
  """

  def generate_subgraph(
      self, model: onnx.ModelProto, input_name: str, output_name: str
  ) -> tuple[
      list[onnx.NodeProto], list[onnx.TensorProto], list[onnx.ValueInfoProto]
  ]:
    """Generates the subgraph to inject and the initializers."""

    graph = graph_builder.GraphBuilder()
    graph.add_input(input_name)
    graph.add_output(output_name)
    output = reroute_subgraph(graph, input_name, 0, 1)
    graph.add_edge(output, output_name)

    new_nodes, initializers, new_outputs = graph.build()

    return new_nodes, initializers, new_outputs
