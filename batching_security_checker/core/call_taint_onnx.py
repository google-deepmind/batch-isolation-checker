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

"""Convert ONNX model into jax function."""

import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

from jax import numpy as jnp
from jax.experimental import checkify
from jaxonnxruntime.core import config_class
from jaxonnxruntime.core import handler as onnx_handler
from jaxonnxruntime.core import onnx_graph
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_utils
import rich
from rich import progress

from batching_security_checker.core import report
from batching_security_checker.core import taint_handler
from batching_security_checker.core import taint_propagation
import onnx
from onnx import helper as onnx_helper


config = config_class.config
OnnxNode = onnx_node.OnnxNode
OnnxGraph = onnx_graph.OnnxGraph
Handler = onnx_handler.Handler

logger = logging.getLogger(__name__)


def _get_dtype(graph: OnnxGraph, tensor_name: str) -> jnp.dtype:
  if graph.value_info_dict.get(tensor_name) is not None:
    tensor_proto = graph.value_info_dict[tensor_name]
    return onnx_utils.tensor_dtype_to_jnp_dtype(
        tensor_proto.type.tensor_type.elem_type
    )
  elif tensor_name in graph.get_constant_dict():
    tensor = graph.get_constant_dict()[tensor_name]
    return tensor.dtype
  else:
    raise ValueError(
        f'{tensor_name=} not found in graph: you may need to run'
        ' shape inference on the model first: (e.g.,'
        ' onnx.shape_inference.infer_shapes(model))'
    )


def call_onnx_model(
    model: onnx.ModelProto,
    tainted_inputs: Union[Sequence[Any], Dict[str, Any]],
    data_inputs: Optional[Dict[str, Any]] = None,
    rename_tensors: bool = False,
    info_level: int = 2,
) -> Tuple[
    list[report.ProofReport.Info],
    Dict[str, Any],
]:
  """Runs label propagation on an ONNX ModelProto in JAX.

  Args:
    model: The ONNX model to run label propagation on.
    tainted_inputs: The inputs annotated with taint.
    data_inputs: The inputs to the model.
    rename_tensors: If True, renames all onnx.TensorProto names with unique IDs.
    info_level: The logging level.

  Returns:
    A tuple containing:
      - A list of report.ProofReport.Info objects to indicate violations.
      - The tainted outputs of the model.
  """

  graph = model.graph
  if rename_tensors:
    logging.info(
        'In call_onnx_model: rename the onnx tensors with unique_id'
        ' `tensor_{id}.'
    )
    graph = onnx_utils.sanitize_tensor_names_in_graph(graph)
  graph_helper = OnnxGraph(graph)
  if model.ir_version < 3:
    opset = [onnx_helper.make_opsetid(onnx.defs.ONNX_DOMAIN, 1)]
  else:
    opset = model.opset_import

  # NOTE: Add a custom domain because of SimplifiedLayerNormalization.
  # (later we may replace certain operators with custom ones from this opset)
  opset.append(onnx.OperatorSetIdProto(domain='experimental', version=1))

  model_params = graph_helper.initializer_dict
  input_names = onnx_utils.get_graph_input(graph)

  # TODO: Rather than adding all of them before running the model,
  # we could add them as we need them for the first time (the same fortaints)
  tensor_dict = dict(
      # **onnx_utils.maybe_convert_to_dict(tainted_inputs, input_names),
      **model_params
  )

  if data_inputs is not None:
    for name, tensor in data_inputs.items():
      if name in tensor_dict:
        raise ValueError(
            f'Cannot set data_inputs for {name} because it is already in the'
            ' model_params'
        )
      tensor_dict[name] = tensor

  tainted_model_params = {}
  for name, tensor in model_params.items():
    tainted_model_params[name] = jnp.full_like(
        tensor,
        taint_propagation.identity_element(),
        dtype=taint_propagation.tdtype().jnp_full,
    )

  tainted_tensor_dict = dict(
      **onnx_utils.maybe_convert_to_dict(tainted_inputs, input_names),
      **tainted_model_params,
  )

  for name, tensor in tainted_tensor_dict.items():
    if name not in tensor_dict:
      dtype = _get_dtype(graph_helper, name)
      tensor_dict[name] = jnp.full_like(tensor, jnp.nan, dtype=dtype)

  taint_violations, graph_outputs = call_onnx_graph(
      graph,
      tensor_dict,
      tainted_tensor_dict,
      opset=opset,
      info_level=info_level,
  )
  # del tensor_dict
  # del tainted_tensor_dict

  return taint_violations, graph_outputs


def _find_output_node(tensor_name: str, graph: onnx.GraphProto):
  for node in graph.node:
    if tensor_name in node.output:
      return (node.op_type, node.name)
  return None

def call_onnx_graph(
    graph: onnx.GraphProto,
    tensor_dict: Dict[str, Any],  # public inputs to the model e.g.,
    taint_tensor_dict: Dict[str, Any],
    opset: ... = None,
    info_level: int = 2,
) -> Tuple[list[report.ProofReport.Info], Dict[str, Any]]:
  """Convert ONNX.GraphProto to jax_func with ONNX.GraphProto.initializer as parameters."""

  tensor_ref_dict = build_ref_dict(graph)
  graph_helper = OnnxGraph(graph)

  taint_violation_infos = []

  # step 1: Trace those static info
  jit_func_dict = {}
  taint_jit_func_dict = {}
  onnx_node_dict = {}
  if opset is None:
    opset = [
        onnx_helper.make_opsetid(
            onnx.defs.ONNX_DOMAIN, onnx.defs.onnx_opset_version()
        )
    ]
  handlers = _get_all_handlers(opset)
  node_execute_order_list = graph_helper.topological_sort()

  ref_dict = {}

  # logger.info('Start tracing the jax_func model to get some static info')
  for node_proto in progress.track(
      node_execute_order_list,
      total=len(node_execute_order_list),
      description='Tracing label propagation...',
      disable=info_level < 1,
  ):
    node = OnnxNode(node_proto, graph_helper)
    onnx_node_dict[node.name] = node

    # check the correrspondence of tensors in tensor_dict and taint_tensor_dict
    assert set(taint_tensor_dict.keys()) == set(tensor_dict.keys()), (
        f'taint_tensor_dict keys = {taint_tensor_dict.keys()}, tensor_dict'
        f' keys = {tensor_dict.keys()}'
    )
    for name, data_tensor in tensor_dict.items():
      taint_tensor = taint_tensor_dict[name]

      if data_tensor.shape != taint_tensor.shape:
        raise ValueError(f'{name=} {data_tensor.shape=} != {taint_tensor.shape=} @{_find_output_node(name, graph)}')

      assert taint_tensor.dtype == taint_propagation.tdtype().jnp_full, (
          f'{name=} {taint_tensor.dtype=} !='
          f' {taint_propagation.tdtype().jnp_full}'
      )

    for x in node.inputs + node.subgraph_inputs:
      # For those optional input arguments defined by the ONNX op, if there is
      # no user input, we will manually set these args to be `None`.
      if x and x not in taint_tensor_dict:
        raise ValueError(
            f'Fail to get the input tensor {x} of node input names'
            f'{node.inputs + node.subgraph_inputs}, the node proto is\n'
            f'{node.node_proto}.'
            f'tainted_tensor_dict keys = {taint_tensor_dict.keys()}'
        )

      if x and x not in tensor_dict:
        raise ValueError(
            f'Fail to get the input tensor {x} of node input names'
            f'{node.inputs + node.subgraph_inputs}, the node proto is\n'
            f'{node.node_proto}.'
            f'tainted_tensor_dict keys = {tensor_dict.keys()}'
        )

    node_data_inputs = [
        tensor_dict[x] if x else None
        for x in node.inputs + node.subgraph_inputs
    ]

    node_taint_inputs = [
        taint_tensor_dict[x] if x else None
        for x in node.inputs + node.subgraph_inputs
    ]

    data_jit_func, taint_jit_func, taint_attrs_dict = _get_jit_funcs(
        node, node_data_inputs, node_taint_inputs, handlers=handlers
    )
    taint_jit_func_dict[node.name] = taint_jit_func

    if info_level >= 2:
      _info_pre_check(node, node_taint_inputs)

    n_violations_prev = len(taint_violation_infos)

    err, taint_out = checkify.checkify(taint_jit_func)(
        *node_data_inputs, *node_taint_inputs, **taint_attrs_dict
    )
    err.throw()

    taint_out = taint_out if isinstance(taint_out, Sequence) else [taint_out]

    for name, output in zip(node.outputs, taint_out):

      if output is None:
        continue

      if not taint_propagation.has_all_single_taint(output):
        colors = taint_propagation.from_taint_to_unique_interfering_colors(
            output
        )
        info = report.ProofReport.Info(
            type='label_interference',
            location=node.name,
            tensor_name=name,
            tensor_shape=output.shape,
            colors=colors,
        )

        taint_violation_infos.append(info)

      taint_tensor_dict[name] = output

    if data_jit_func:
      jit_func_dict[node.name] = data_jit_func
      data_out = data_jit_func(*node_data_inputs, **node.attrs_dict)
      data_out = data_out if isinstance(data_out, Sequence) else [data_out]

      for name, output, taint_output in zip(node.outputs, data_out, taint_out):
        if output is None:
          continue
        assert jnp.isdtype(taint_output, taint_propagation.tdtype().jnp_full), (
            f'{name=} {jnp.dtype(taint_output)=} !='
            f' {taint_propagation.tdtype().jnp_full}'
        )

        expected_data_dtype = _get_dtype(graph_helper, tensor_name=name)
        assert jnp.isdtype(
            output, expected_data_dtype
        ), f'{name=} {output.dtype=} != {expected_data_dtype=}'

        if jnp.all(taint_propagation.is_all_untainted(taint_output)):
          tensor_dict[name] = output
        else:
          # TODO: We could reduce memory if we don't need to keep around the nan tensors and instead can say that they are e.g., None
          tensor_dict[name] = jnp.full_like(output, jnp.nan, dtype=None)

    else:
      for name, output in zip(node.outputs, taint_out):
        # TODO: We could reduce memory if we don't need to keep around the nan tensors and instead can say that they are e.g., None
        tensor_dict[name] = jnp.full_like(output, jnp.nan, dtype=None)

    if info_level >= 2:
      _info_post_check(
          node, taint_out, taint_violation_infos[n_violations_prev:]
      )

    # `tensor_ref_dict` counts how many times a tensor is used as an input.
    # Once it was used this many times as an input, we can delete the tensor.
    for input_ in node.inputs + node.subgraph_inputs:
      if input_ in ref_dict:
        ref_dict[input_] += 1
      else:
        ref_dict[input_] = 1
    remove_keys = []
    for k, v in ref_dict.items():
      if tensor_ref_dict[k] == v:
        remove_keys.append(k)
    for rm_k in remove_keys:
      if not rm_k:
        continue
      del ref_dict[rm_k]
      del tensor_dict[rm_k]
      del taint_tensor_dict[rm_k]

  graph_outputs = {n.name: taint_tensor_dict[n.name] for n in graph.output}
  return taint_violation_infos, graph_outputs


def build_ref_dict(graph: onnx.GraphProto) -> Dict[str, int]:
  """Initialize reference count dict."""
  ref_dict: dict[Any, Any] = {}
  for node in graph.node:
    if onnx_utils.contain_subgraph(node):
      for a in node.attribute:
        if a.HasField('g'):
          sub_ref_dict = build_ref_dict(a.g)
          ref_dict.update(
              {k: ref_dict.get(k, 0) + v for k, v in sub_ref_dict.items()}
          )
    inputs = node.input
    for input_ in inputs:
      if input_ in ref_dict:
        ref_dict[input_] += 1
      else:
        ref_dict[input_] = 1
  for o in graph.output:
    ref_dict[o.name] = ref_dict[o.name] + 1 if o.name in ref_dict else 1
  return ref_dict


def _get_all_handlers(
    opset: Sequence[onnx.OperatorSetIdProto],
) -> Dict[str, Dict[str, Type[taint_handler.TaintHandler]]]:
  """Get all ONNX OP_TYPE handlers from Handler subclasses.

  Args:
      opset: An OperatorSetIdProto message containing the operator set version
        information.

  Returns:
      A dictionary of all the ONNX handlers, where the keys are the domain
      names
      and the values are nested dictionaries mapping operator names to their
      Handler
      subclasses.

  Raises:
      ValueError: If there is no OP_TYPE attribute defined in the Handler class.
  """

  handlers: Dict[Any, Any] = {}

  for thandler in taint_handler.TaintHandler.__subclasses__():

    handler = thandler.data_handler()

    if not hasattr(handler, 'OP_TYPE'):
      logger.warning(
          (
              "%s doesn't have ONNX OP_TYPE. "
              'Please use handler.register_op decorator to register it.'
          ),
          handler.__name__,
      )

    domain = handler.DOMAIN
    opset_dict = dict([(o.domain, o.version) for o in opset])
    if handler.DOMAIN not in opset_dict:
      # logging.debug(
      #    'handler.DOMAIN %s is not in opset_dict %s, skip it.',
      #    domain,
      #    opset_dict,
      # )
      continue
    version = opset_dict[handler.DOMAIN]
    since_version = handler.get_since_version(version)

    if since_version > 0:
      handler.SINCE_VERSION = since_version
      handlers.setdefault(domain, {})[handler.OP_TYPE] = thandler
  return handlers


def _get_jit_funcs(
    node: OnnxNode,
    data_inputs: list[Any],
    taint_inputs: list[Any],
    handlers: Dict[str, Dict[str, type[taint_handler.TaintHandler]]],
    **kwargs,
):
  """Get the JAX node implementation."""

  handler = (
      handlers[node.domain].get(node.op_type, None)
      if node.domain in handlers
      else None
  )
  if handler:

    data_jit_func, taint_jit_func = handler.handle_taint(
        node, data_inputs, taint_inputs, **kwargs
    )
    taint_attrs_dict = handler.prepare_taint_attrs(node.attrs_dict)
    return data_jit_func, taint_jit_func, taint_attrs_dict
  else:
    raise NotImplementedError(f'{node.op_type} is not implemented.')


def _info_pre_check(node: OnnxNode, node_taint_inputs):
  """Print infos about the node to be checked."""

  rich.print('----------------------------------------------------------------')
  rich.print(f'Checking Operator [bold magenta]{node.op_type}[/bold magenta]:')
  rich.print(f'  Name: {node.name}')
  rich.print(f'  Attributes: {node.attrs_dict}')
  inputs = node.inputs + node.subgraph_inputs
  inputs = {
      k: v.shape if v is not None else None
      for k, v in zip(inputs, node_taint_inputs)
  }
  rich.print(f'  Inputs: {inputs}')


def _info_post_check(
    node: OnnxNode,
    node_taint_outputs,
    taint_violation_infos: list[report.ProofReport.Info],
):
  """Print infos about the node check."""

  outputs = {
      k: v.shape if v is not None else None
      for k, v in zip(node.outputs, node_taint_outputs)
  }
  rich.print(f'  Outputs: {outputs}')

  if not taint_violation_infos:
    rich.print(
        '-> Op check complete :white_check_mark:, no interference found.'
    )
  else:
    rich.print(
        '-> Op check failed :cross_mark:, found possible interference'
        ' in output:'
    )
    for info in taint_violation_infos:
      rich.print(f'  {info.tensor_name}: {info.type}')
  rich.print('----------------------------------------------------------------')
