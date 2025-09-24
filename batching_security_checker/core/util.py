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

"""Utility functions for model checker."""

import glob
import os


def resolve_model_path(model_dir: str, model: str):
  """Resolves the path to a model file."""

  model_path = os.path.join(model_dir, f"{model}.onnx")
  model_dir = os.path.join(model_dir, model)

  if os.path.isfile(model_path):
    # can provide a single model file
    pass
  elif os.path.isdir(model_dir):
    # can provide a directory if there is only one model in it
    models = glob.glob(os.path.join(model_dir, "*.onnx"))
    if len(models) != 1:
      raise ValueError(f"Expected one model in {model_dir}, but found {models}")
    model_path = models[0]
  else:
    raise ValueError(f"Model {model} does not exist in {model_dir}")

  return model_path
