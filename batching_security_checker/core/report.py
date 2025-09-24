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

"""Create a model checker proof report."""

import dataclasses
from typing import Any, Literal
import attrs as attr


#@dataclasses.dataclass(kw_only=True)
@attr.define
class ProofReport:
  """A report of a model checker proof."""

  #@dataclasses.dataclass(kw_only=True)
  @attr.define
  class Info:
    type: Literal["unexpected_color", "label_interference"]
    location: str
    tensor_name: str
    tensor_shape: tuple[int, ...]
    colors: list[
        dict[str, Any]
    ]  # {"expected": None, "actual": None} or {"min": None, "max": None}

  ir_version: int
  model_version: int
  domain: str
  graph_name: str
  concrete_dim_shape: dict[str, int]
  proof_status: Literal["PASSED", "FAILED"]
  infos: list[Info]
