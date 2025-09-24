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

"""Loads onnx models from the github repo onnx/models."""

import glob
import os
import re
import subprocess
from typing import Any
import warnings

import onnx
from rich import progress

from batching_security_checker.core import models


def get_models(
    base_models_dir: str, repo_subdir: str = "", try_load_models: bool = False
) -> list[models.Model]:
  """Loads all onnx models from the github repo."""

  repo_dir = os.path.join(base_models_dir, "models")
  if not os.path.isdir(repo_dir):
    print(f"Cloning repo to {repo_dir}...")
    subprocess.run(
        "git clone https://github.com/onnx/models.git",
        shell=True,
        capture_output=True,
        text=True,
        check=True,
        cwd=base_models_dir,
    )

  print(
      f"Searching for onnx models from {repo_dir} (subdir: '{repo_subdir}')..."
  )
  model_dict = _get_all_onnx_models(repo_dir, repo_subdir)
  model_dict = _set_model_info(model_dict, repo_dir)
  n_models = sum(len(x) for x in model_dict.values())
  print(f"-> Found {n_models} models.")

  print("Selecting one model per opset version...")
  models_selected = _select_models(model_dict)
  print(f"-> Selected {len(models_selected)} models")
  models_selected = _download_models(models_selected, repo_dir)
  print(f"-> There are {len(models_selected)} models available locally")

  if try_load_models:
    for model in progress.track(
        models_selected,
        total=len(models_selected),
        description="Check loading models...",
    ):
      onnx.load(os.path.join(repo_dir, model["path"]))

  models_lst = []
  for model in models_selected:
    model = models.Model(
        name=model["path"],
        src=models.LocalSrc(path=os.path.join(repo_dir, model["path"])),
        labeler=None,
        dim_params="auto",
    )
    models_lst.append(model)

  return models_lst


def _get_all_onnx_models(
    repo_dir: str, repo_subdir: str
) -> dict[str, list[dict[str, Any]]]:
  """Find all onnx models in the given repo."""

  search_path = os.path.join(repo_dir, repo_subdir, "**/*.onnx")

  files = glob.glob(
      search_path,
      recursive=True,
  )

  model_dict = {}
  for file in files:
    file_path = file.removeprefix(repo_dir + "/")

    # extract opset version
    match = re.search(r"Opset(\d+)", file_path)
    if match:
      opset_version = int(match.group(1))
    else:
      opset_version = None

    # obtain opset invariant model path
    opset_replaced_path = re.sub(r"Opset\d+", "", file_path)
    opset_replaced_path = opset_replaced_path.strip("_")

    if opset_replaced_path not in model_dict:
      model_dict[opset_replaced_path] = []
    model_dict[opset_replaced_path].append(
        {"path": file_path, "opset_version": opset_version}
    )
  return model_dict


def _get_lfs_model_sizes(repo_path: str) -> dict[str, float]:
  """Get the size of all models managed by lfs (not yet downloaded)."""

  process = subprocess.run(
      "git lfs ls-files -s",
      shell=True,
      capture_output=True,
      text=True,
      check=True,
      cwd=repo_path,
  )
  output = process.stdout

  model_sizes = {}
  for line in output.splitlines():
    match = re.match(r"[a-z0-9]+ - (.+.onnx) \(([\d.]+\s*[KMGT]?B)\)", line)
    if match:
      file_path = match.group(1).strip()
      size_str = match.group(2)

      m = re.search(r"([\d.]+)\s*([KMGT]?B)?", size_str)
      if not m:
        raise ValueError(f"Invalid size string: {size_str}")
      size = float(m.group(1))
      unit = m.group(2)

      if unit == "KB":
        size *= 1024
      elif unit == "MB":
        size *= 1024**2
      elif unit == "GB":
        size *= 1024**3
      elif unit == "TB":
        size *= 1024**4

      model_sizes[file_path] = size
  return model_sizes


def _set_model_info(
    model_dict: dict[str, list[dict[str, Any]]], repo_path: str
):
  """Updates the model_dict with the model size and whether the model is local or not."""

  models_lfs = _get_lfs_model_sizes(repo_path)

  for _, variants in model_dict.items():
    for variant in variants:
      local_file_size = os.path.getsize(
          os.path.join(repo_path, variant["path"])
      )
      if variant["path"] in models_lfs:
        # model may not yet be downloaded
        variant["size_bytes"] = models_lfs[variant["path"]]

        if local_file_size <= 200 and variant["size_bytes"] > 200:
          variant["is_local"] = False
        else:
          warnings.warn(
              f"Using local version of model {variant['path']}; skipping"
              " download from LFS."
          )
          variant["is_local"] = True
      else:
        variant["size_bytes"] = local_file_size
        variant["is_local"] = True

  return model_dict


def _format_human_readable_size(size_bytes: float) -> str:
  """Formats a size in bytes as a human readable string."""
  for unit in ["bytes", "KB", "MB", "GB", "TB"]:
    if size_bytes < 1024:
      return f"{size_bytes:.2f} {unit}"
    size_bytes /= 1024
  raise ValueError(f"Size {size_bytes} cannot be formatted.")


def _select_models(
    model_dict: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
  """Selects the model with the highest opset version for each model name."""

  model_paths = []
  for _, lst in model_dict.items():
    lst = sorted(lst, key=lambda x: x["opset_version"], reverse=True)
    selected_model = lst[0]
    model_paths.append(selected_model)
  return model_paths


def _download_models(
    model_lst: list[dict[str, Any]], repo_path: str
) -> list[dict[str, Any]]:
  """Downloads all models that are not yet available locally."""

  to_download_bytes = 0
  local_lst = []
  download_lst = []
  for model in model_lst:
    if model["is_local"]:
      local_lst.append(model)
    else:
      to_download_bytes += model["size_bytes"]
      download_lst.append(model)

  total_size_str = _format_human_readable_size(to_download_bytes)

  print(
      f"There are {len(local_lst)}/{len(model_lst)} models available locally."
  )

  if not download_lst:
    return local_lst
  elif to_download_bytes > 500_000_000:
    prompt = (
        f"Do you want to download all {len(download_lst)} unavailable models?"
        f" Requires {total_size_str} space. ([y], n)"
    )

    while True:
      user_input = input(prompt).lower()

      if not user_input:  # enter
        user_input = "y"

      if user_input in ("y", "yes"):
        break
      elif user_input in ("n", "no"):
        return local_lst
      else:
        print("Invalid input. Please enter 'y' or 'n'.")

  # git lfs prune
  n_successes = 0
  n_failed = 0
  for model in progress.track(
      download_lst,
      total=len(download_lst),
      description="Downloading models...",
  ):

    model_path = model["path"]
    print(f"Downloading {model_path}...")
    try:
      subprocess.run(
          f'git lfs pull --include="{model_path}" --exclude=""',
          shell=True,
          capture_output=True,
          text=True,
          check=True,
          cwd=repo_path,
      )
      local_lst.append(model)
      n_successes += 1
    except subprocess.CalledProcessError as e:
      print(f"Failed to download {model_path}: {e}")
      n_failed += 1

  print(
      f"Downloaded {n_successes} models, failed to download {n_failed} models"
  )

  return local_lst
