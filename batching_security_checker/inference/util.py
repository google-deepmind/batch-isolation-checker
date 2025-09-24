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

"""Utility for running inference."""

from jaxonnxruntime.core import onnx_utils
import onnx


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
        raise ValueError(f"Unknown dim type: {dim}")
    shape = tuple(shape)
    return shape

  for i, proto in enumerate(model.graph.input):
    dtype = onnx_utils.tensor_dtype_to_jnp_dtype(
        proto.type.tensor_type.elem_type
    )
    shape = _get_shape(proto, dim_params)
    inputs[proto.name] = {"shape": shape, "dtype": dtype, "idx": i}

  for i, proto in enumerate(model.graph.output):
    dtype = onnx_utils.tensor_dtype_to_jnp_dtype(
        proto.type.tensor_type.elem_type
    )
    shape = _get_shape(proto, dim_params)
    outputs[proto.name] = {"shape": shape, "dtype": dtype, "idx": i}

  return dim_params, inputs, outputs
