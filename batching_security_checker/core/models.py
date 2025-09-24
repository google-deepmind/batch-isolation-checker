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

"""Representations of models that can be checked."""

from typing import List, Literal
import warnings

import attrs as attr
import cattr
import onnx
import os

from batching_security_checker.core import labels
from batching_security_checker.core import model_summary


@attr.define
class LocalSrc:
  path: str
  type: Literal["local"] = "local"


@attr.define
class GitHubSrc:
  repo: str
  owner: str
  path: str
  branch: str = "main"
  commit: str | None = None
  type: Literal["github"] = "github"


@attr.define
class HuggingfaceSrc:
  model_id: str
  revision: str | None = None
  type: Literal["huggingface"] = "huggingface"


@attr.define
class ModelFixer:

  def fix_model(self, model: onnx.ModelProto) -> bool:
    is_changed = False
    for node in model.graph.node:
      if not node.domain and node.op_type == "SimplifiedLayerNormalization":
        node.domain = "experimental"
        warnings.warn(
            "Setting domain of SimplifiedLayerNormalization to experimental"
        )
        is_changed = True
    if is_changed:
      exp_opset = onnx.OperatorSetIdProto()
      exp_opset.version = 1
      exp_opset.domain = "experimental"
      model.opset_import.append(exp_opset)
    return is_changed


@attr.define
class AutoDimParams:
  """Default values for dim params based on their name, and if undefined."""

  default_by_name: dict[str, int]
  default: 4

  def assign_values(self, dim_params: list[str]) -> dict[str, int]:
    return {dim_param: self.default_by_name.get(dim_param, self.default) for dim_param in dim_params}


@attr.define
class Model:
  """Represents a model that can be checked."""

  name: str
  src: LocalSrc | GitHubSrc | HuggingfaceSrc

  # NOTE: could add a custom labeler here
  labeler: labels.BatchDimLabeler | None = None

  dim_params: Literal["auto"] | List[dict[str, int]] | AutoDimParams | None = None

  model_fixer: ModelFixer | None = ModelFixer()
  _onnx_model: onnx.ModelProto | None = attr.field(metadata={"exclude": True}, default=None)

  @classmethod
  def converter(cls):
    c = cattr.Converter()
    unst_hook = cattr.gen.make_dict_unstructure_fn(
        Model, c, _onnx_model=cattr.override(omit=True)
    )
    c.register_unstructure_hook(Model, unst_hook)
    return c

  def load(self, force: bool = False, check: bool = False) -> onnx.ModelProto:
    if self._onnx_model is None or force:
      self.download()
      model_path = self.get_path()



      print(f"Loading model from {model_path}")
      onnx_model = onnx.load(model_path)


      if self.model_fixer is not None:
        is_changed = self.model_fixer.fix_model(onnx_model)

        if is_changed:

          tmp_model_path = "/tmp/model.onnx"
          tmp_model_data_name = f"{os.path.basename(tmp_model_path)}.data"
          os.makedirs(os.path.dirname(tmp_model_path), exist_ok=True)

          onnx.save(
            onnx_model,
            tmp_model_path,
            save_as_external_data=True,
            location=tmp_model_data_name,
          )

          # after saving model needs to be reloaded (old onnx_model_updated cannot be used anymore)
          onnx_model = onnx.load(tmp_model_path)
          model_path = tmp_model_path

      if check:
        onnx.checker.check_model(model_path, check_custom_domain=False)

      self._onnx_model = onnx_model
    return self._onnx_model

  def get_dim_params(self) -> List[dict[str, int]]:
    if self.dim_params == "auto":
      # hardcode all dim params to 4
      self.dim_params = AutoDimParams(default_by_name={}, default=4)

    if isinstance(self.dim_params, AutoDimParams):
      assert self._onnx_model is not None
      dim_params, _, _ = model_summary.model_inputs_outputs(self._onnx_model)
      self.dim_params =  [self.dim_params.assign_values(dim_params)]

    return self.dim_params

  def get_path(self) -> str:
    """Returns the file path to the model dir."""
    if isinstance(self.src, LocalSrc):
      return self.src.path
    else:
      # TODO: Determine a local path for the model where it can be downloaded
      raise ValueError("Get path not implemented for {type(self.src)}")

  def download(self, force: bool = False):
    # TODO: ensure downloaded if not existing
    pass

  def clear_model(self):
    self._onnx_model = None
