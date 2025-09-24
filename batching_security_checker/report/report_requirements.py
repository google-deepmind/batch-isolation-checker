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

"""Reports on the requirements for the model checker."""

import json
from typing import Any, Sequence

import rich
from rich import tree as rich_tree


def report_cli(reports_json: str):
  """Reports on the requirements for the model checker."""

  options = create_report(reports_json)
  rich.print(options[0][1])

  width = 50

  while True:

    n = (width - len("Options")) // 2
    print(f"\n{n * '='} Options {n * '='}")
    for i, (name, _) in enumerate(options):
      print(f"{i+1}. {name}")

    print(f"{len(options)+1}. Exit")

    try:
      choice = input(f"Enter your choice (1-{len(options)+1}): ")
      choice = int(choice)

      if choice > 0 and choice <= len(options):
        name, data = options[choice - 1]

        n = (width - len(name)) // 2
        print(f"\n{n * '='} {name} {n * '='}")
        if isinstance(data, list):
          for d in data:
            rich.print(d)
            print()
        else:
          rich.print(data)
        print(width * "=")
      elif choice == len(options) + 1:
        print("Exiting...")
        break
      else:
        print(
            "Invalid choice. Please enter a number between 1 and"
            f" {len(options)+1}."
        )

    except ValueError:
      print("Invalid input. Please enter a number.")

    # print("\n===================================")


def create_report(reports_json: str):
  """Summarizes the reports."""

  with open(reports_json, "r") as f:
    reports = json.load(f)

  # 1. Aggregate Reports
  def _aggregate_ops(
      accumulator: dict[str, dict[str, int]],
      missing_ops: dict[str, Sequence[str]] | None,
  ):
    has_missing_ops = False
    if missing_ops is None:
      return has_missing_ops
    for domain, ops in missing_ops.items():
      if domain not in accumulator:
        accumulator[domain] = dict()
      for op in set(ops):
        if op not in accumulator[domain]:
          accumulator[domain][op] = 0
        accumulator[domain][op] += 1
        has_missing_ops = True
    return has_missing_ops

  missing_taint_ops = dict()
  missing_data_ops = dict()
  errors = dict()
  non_fixed_models = []
  fixed_models = []

  all_ops_models = []
  n_taint_ops_missing = 0
  n_data_ops_missing = 0

  for report in reports:
    # TODO: potentially deal with dataclass

    # if isinstance(report, RequirementsReportError):
    #  errors[report.model.name] = '\n'.join(report.errors)
    #  continue
    # elif isinstance(report, RequirementsReport):
    if "errors" in report:
      errors[report["model"]["name"]] = "\n".join(report["errors"])
      continue

    elif "missing_taint_ops" in report:
      if report["is_fixed"]:
        fixed_models.append(report)
        has_missing_taint_ops = _aggregate_ops(
            missing_taint_ops, report["missing_taint_ops"]
        )
        has_missing_data_ops = _aggregate_ops(
            missing_data_ops, report["missing_data_ops"]
        )
        n_taint_ops_missing += int(has_missing_taint_ops)
        n_data_ops_missing += int(has_missing_data_ops)
        if (not has_missing_taint_ops) and (not has_missing_data_ops):
          all_ops_models.append(report["model"])

      else:
        non_fixed_models.append(report)
    else:
      raise ValueError(f"Unknown report type: {type(report)}")

  # TODO: Show errors + get insights into non-fixed models

  # 2. Create overview of requirements
  overview_tree = create_overview(
      n_total=len(reports),
      n_errors=len(errors),
      n_dynamic=len(non_fixed_models),
      n_fixed=len(fixed_models),
      n_all_ops=len(all_ops_models),
      n_taint_ops_missing=n_taint_ops_missing,
      n_data_ops_missing=n_data_ops_missing,
  )

  # 3. Suggest priority of missing operators
  taint_ops_tree = compute_missing_operator_priority(
      fixed_models, missing_taint_ops, "missing_taint_ops"
  )

  data_ops_tree = compute_missing_operator_priority(
      fixed_models, missing_data_ops, "missing_data_ops"
  )

  models_error = sorted([x.strip() for x in errors.keys()])
  models_non_fixed = sorted(
      [x["model"]["name"].strip() for x in non_fixed_models]
  )
  models_supported = sorted([x["name"].strip() for x in all_ops_models])

  options = [
      ("Overview", overview_tree),
      ("Missing Operators", [taint_ops_tree, data_ops_tree]),
      ("Models with Errors", errors),
      ("Model Ids with Errors", models_error),
      ("Models with Dynamic Shapes", non_fixed_models),
      ("Model Ids with Dynamic Shapes", models_non_fixed),
      ("Supported Models", all_ops_models),
      ("Supported Model Ids", models_supported),
  ]
  return options


def create_overview(
    n_total: int,
    n_fixed: int,
    n_dynamic: int,
    n_errors: int,
    n_all_ops: int,
    n_taint_ops_missing: int,
    n_data_ops_missing: int,
) -> rich_tree.Tree:
  """Creates an overview of the requirements."""

  assert n_total == n_fixed + n_dynamic + n_errors, f"{n_total=}  {n_fixed=}   {n_dynamic=}   {n_errors=}"
  assert n_all_ops <= n_fixed
  assert n_taint_ops_missing <= n_fixed - n_all_ops
  assert n_data_ops_missing <= n_fixed - n_all_ops

  tree = rich_tree.Tree(f"Total Models Checked: {n_total}")

  tree.add(f"Skipped models due to errors: {n_errors}")
  tree.add(f"Unsupported models due to dynamic shapes: {n_dynamic}")
  fixed_models = tree.add(f"Fixed models: {n_fixed} (candidates)")

  fixed_models.add(
      f"Models with all operators implemented: {n_all_ops} (supported)"
  )
  missing_ops = fixed_models.add(
      f"Models with missing operators: {n_fixed - n_all_ops}"
  )

  missing_ops.add(f"Models with missing label operators: {n_taint_ops_missing}")
  missing_ops.add(f"Models with missing data operators: {n_data_ops_missing}")

  return tree


def compute_missing_operator_priority(
    results: list[dict[str, Any]],
    operators: dict[str, dict[str, int]],
    op_type: str,
):
  """Suggests the order in which to implement missing operators."""

  ops_lst = []
  for domain, ops in operators.items():
    for op, count in ops.items():
      ops_lst.append(
          {"id": f"{domain}.{op}", "domain": domain, "op": op, "count": count}
      )

  ops_lst = sorted(ops_lst, key=lambda x: x["count"], reverse=True)

  additional_ops = dict()
  cur_supported_models = _find_supported_models(
      results, additional_ops, op_type
  )
  n_supported = len(cur_supported_models)

  for x in ops_lst:
    op = x["op"]
    domain = x["domain"]

    # add one more op to additional ops
    if domain not in additional_ops:
      additional_ops[domain] = []
    additional_ops[domain].append(op)

    # find the supported models with this additional operator
    supported_models = _find_supported_models(results, additional_ops, op_type)

    x["n_additonal_cum"] = len(supported_models) - n_supported
    x["n_supported_cum"] = len(supported_models)
    n_supported = len(supported_models)

  # format output
  op_type_lookup = {
      "missing_taint_ops": "LABEL",
      "missing_data_ops": "DATA",
  }

  tree = rich_tree.Tree(
      f"We currently have the {op_type_lookup[op_type]} operators implemented"
      f" for {len(cur_supported_models)}/{len(results)} models. The following"
      " implementation order will provide the fastest increase in model"
      " coverage:"
  )

  for op in ops_lst:
    tree.add(
        f"{op[ 'id']}: +{op['n_additonal_cum']}   (total:"
        f" {op['n_supported_cum']}/{len(results)})"
    )

  return tree


def _get_ops(operators: dict[str, Sequence[str]]) -> set[str]:
  """Converts operators to a set of strings."""
  names = set()
  for domain, ops in operators.items():
    for op in ops:
      names.add(f"{domain}.{op}")
  return names


def _find_supported_models(
    results: list[dict[str, Any]],
    additional_ops: dict[str, Sequence[str]],
    op_type: str,
) -> list[str]:
  """Finds the models that are supported with the additional operators."""

  additional_set = _get_ops(additional_ops)

  supported_models = []

  for report in results:
    missing_set = _get_ops(report[op_type])

    # print(f"{op_type}:     {additional_set=}    {missing_set=}")

    if not (missing_set - additional_set):
      # no more missing with additioal
      supported_models.append(report["model"]["name"])

  return supported_models
