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

import os
import tempfile
import time
import warnings
from collections.abc import Sequence
from typing import (Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple,
                    Union)

import attrs as attr
import cattrs
import onnx
import tinydb
from onnxruntime.tools import symbolic_shape_infer
from rich import progress
import rich
from rich import prompt
from rich import tree as rich_tree

from jaxonnxruntime.core import onnx_graph
#from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.core import onnx_utils

from batching_security_checker import backend as jax_taint_backend
from batching_security_checker import check_interference
from batching_security_checker.core import (labels, model_summary, models,
                                            report, util)
from batching_security_checker.models import huggingface_onnx

import jax
from jaxonnxruntime import config_class
from batching_security_checker.core import config_class as taint_config


from rich import console # import console.Group
from rich import panel #import panel.Panel
from rich import progress # import progress.Progress, progress.BarColumn, progress.TextColumn, progress.TimeElapsedColumn
from rich import live as rich_live # live.Live





@attr.define
class StageStat:

  stage_name: str
  success_status: str
  status: dict[str, int]  = attr.field(
      default=attr.Factory(dict)
  )



  @property
  def total(self):
    return sum(self.status.values())

  def update(self, status: str):
    if status not in self.status:
      self.status[status] = 0
    self.status[status] += 1


#  def update(self, is_y: bool):
#    if is_y:
#      self.y += 1
#    else:
#      self.n += 1

#  def format(self) -> str:
#    pad = 10
#    pad1 = " " * (pad - len(str(self.n)))
#    pad2 = " " * (pad - len(str(self.y)))
#    return f":x: {self.n}{pad1}:white_check_mark: {self.y}{pad2}(of  {self.total})"

@attr.define
class Stats:
  selection: StageStat = attr.field(factory=lambda: StageStat("Selection Stage", "included") )
  download: StageStat = attr.field(factory=lambda: StageStat("Download Stage", "success"))
  load: StageStat = attr.field(factory=lambda: StageStat("Load Stage", "success"))
  shape_inference: StageStat = attr.field(factory=lambda: StageStat("Shape Inference Stage", "fixed"))
  batch_dim: StageStat = attr.field(factory=lambda: StageStat("Batch Dim Stage", "batch"))
  operator: StageStat = attr.field(factory=lambda: StageStat("Operator Stage", "available"))
  interference_check: StageStat = attr.field(factory=lambda: StageStat("Interference Check Stage", "noninterference"))



  def get_stages(self) -> List[StageStat]:
    return [self.selection, self.download, self.load, self.shape_inference, self.batch_dim, self.operator, self.interference_check]


def _force_recompute(report: "Report", StageCls, recomp_cfg) -> bool:
  stage_cfg: bool | Callable[["Report"], bool] = recomp_cfg.get(StageCls.__name__, False)
  if isinstance(stage_cfg, bool):
    return stage_cfg
  elif isinstance(stage_cfg, Callable):
    return stage_cfg(report)
  raise ValueError(f"Illegal recompute cfg: {recomp_cfg}")


def _build_stage_tree(stats: Stats, cur_stage: int):
    stage_node = None
    tree = rich_tree.Tree(f"[cyan]Model Stages:[/cyan]")
    stages = [stats.selection, stats.download, stats.load, stats.shape_inference, stats.batch_dim, stats.operator, stats.interference_check]
    for i, stat in enumerate(stages):
      if stage_node is None:
        stage = stat.stage_name
        if i == cur_stage:
          stage = f"{stage} [X]"
        else:
          stage = f"{stage} [ ]"
        stage_node = tree.add(f"[bold]{stage}[/bold]")

      for status, count in stat.status.items():
        if status != stat.success_status:
          stage_node.add(f":x: {status}: {count}")

      n_success = stat.status.get(stat.success_status, 0)

      next_stage = stages[i+1].stage_name if i < len(stages) - 1 else "Batching Safe"

      if i + 1 == cur_stage:
        next_stage = f"{next_stage} [X]"
      else:
        next_stage = f"{next_stage} [ ]"

      stage_node = stage_node.add(f":white_check_mark: {stat.success_status}: {n_success}  \n[bold]{next_stage}[/bold]")

    return tree




StagesStr = Literal["SelectionStage", "DownloadStage", "LoadStage", "ShapeInferenceStage", "BatchDimStage", "OperatorStage", "InterferenceCheckStage"]
def create_reports(database_path: str, local_model_dir: str, recompute_cfg: dict[StagesStr, bool | Callable[["Report"], bool]], until_stage: StagesStr = "InterferenceCheckStage", use_bulk_download: bool = True):

  #local_model_dir: str = os.path.join(base_models_dir, "hf-onnx-community")

  force = False # TODO: may want to do more granular and control per stage

  skip_post_download_store = True

  # TODO: Unclear what happens if a stage is force reloaded, should all later stages also be recomputed?

  db = tinydb.TinyDB(database_path)
  report_db = db.table("report")

  dimparam_db = db.table("dimparam")

  exception_db = db.table("exception")

  stats = Stats()

  # TODO: In selection + bulk download record stats

  #### Selection Stage
  rich.print("==========[bold]Starting Selection Stage[/bold]==========")
  rich.print("Select models for the analysis from a different sources (e.g., onnx-community on huggingface).")
  n_reports = len(report_db)
  if n_reports == 0 or _force_recompute(None, SelectionStage, recompute_cfg):
    reports, dim_params_default = SelectionStage.execute(sources=["hf-onnx-community"])
    _update_report_db(reports, report_db=report_db, mode="ReplaceAll")
    dimparam_db.insert_multiple(dim_params_default) # TODO: not quite sure about the update semantics

  if until_stage == SelectionStage.__name__:
    rich.print("==========[bold]Reached final Stage -> Stopping[/bold]==========")
    return


  converter = get_report_converter()
  reports = []
  for r in report_db.all():
    #print()
    #print(r)
    reports.append((r.doc_id, converter.structure(r, Report)))

  #### Download Stage
  rich.print("==========[bold]Starting Download Stage[/bold]==========")
  rich.print("Ensure the selected models are available locally.")
  if use_bulk_download:
    updated_doc_ids = DownloadStage.execute_all(reports, local_dir=local_model_dir)
    updated_doc_ids = set(updated_doc_ids)
    for doc_id, report in progress.track(reports, total=len(reports), description="Updating report database..."):
        if not skip_post_download_store and doc_id in updated_doc_ids:
          _store(report=report, doc_id=doc_id, report_db=report_db)



  main_progress = progress.Progress(
      progress.TextColumn("[bold]Progress:[/bold]"),
      progress.BarColumn(),
      progress.TextColumn("{task.percentage:>3.0f}%"),
      progress.TextColumn(" {task.completed}/{task.total}"),
      progress.TimeElapsedColumn(),
  )
  task_main = main_progress.add_task("main", total=len(reports))

  def _make_panel(cur: int, model_id: str):

    tree = _build_stage_tree(stats, cur)

#    def _active(i: int):
#      return '[X]' if cur == i else '[ ]'

    return panel.Panel(console.Group(
        main_progress,
        f"Current Model: {model_id}",
        tree,
#        f"{_active(0)} Selection Stage:           {stats.selection.format()}",
#        f"{_active(1)} Download Stage:            {stats.download.format()}",
#        f"{_active(2)} Load Stage:                {stats.load.format()}",
#        f"{_active(3)} Shape Inference Stage:     {stats.shape_inference.format()}",
#        f"{_active(4)} Batch Dim Stage:           {stats.batch_dim.format()}",
#        f"{_active(5)} Operator Stage:            {stats.operator.format()}",
#        f"{_active(6)} Interference Check Stage:  {stats.interference_check.format()}",
    ), title="Model Analysis", border_style="cyan")

  with rich_live.Live(_make_panel(0, "-"), refresh_per_second=10) as live:
    for doc_id, report in reports:
      main_progress.update(task_main, advance=1)

      report: Report = report

      updated_onnx_model = None

      # TODO: COMMAND:  poetry run python batching_security_checker/report_cli.py --database /home/ubuntu/batching-security-checker/data/report_db.json --models /home/ubuntu/batching-security-checker/data/hf-onnx-community

      is_selection_success = report.stages.selection is not None and report.stages.selection.status == "included"
      stats.selection.update(report.stages.selection.status)
      live.update(_make_panel(1, report.model_id))
      if until_stage == SelectionStage.__name__ or not is_selection_success:
          continue


      if not use_bulk_download:
        DownloadStage.execute(report, local_dir=local_model_dir)
        _store(report=report, doc_id=doc_id, report_db=report_db)

      is_download_success = report.stages.download is not None and report.stages.download.status == "success"
      stats.download.update(report.stages.download.status)
      live.update(_make_panel(2, report.model_id))
      if until_stage == DownloadStage.__name__ or not is_download_success:
          continue

      #### Load Stage
      #rich.print("==========[bold]Starting Load Stage[/bold]==========")
      #rich.print("Try loading the models and perform a format check.")
      if  is_download_success and (report.stages.load is None or _force_recompute(report, LoadStage, recompute_cfg)):
          LoadStage.clear(report)
          LoadStage.execute(report)
          _store(report=report, doc_id=doc_id, report_db=report_db)

      is_load_success = report.stages.load is not None and report.stages.load.status == "success"
      stats.load.update(report.stages.load.status)
      live.update(_make_panel(3, report.model_id))

      if until_stage == LoadStage.__name__ or not is_load_success:
          continue

      #### Shape Inference Stage
      #rich.print("==========[bold]Starting Shape Inference Stage[/bold]==========")
      #rich.print("Fix dynamic io shapes, run shape inference, and check whether the model has a data independent shape.")
      if is_load_success and (report.stages.shape_inference is None or _force_recompute(report, ShapeInferenceStage, recompute_cfg)):
          dim_params_default = {x["name"]: x["value"] for x in dimparam_db.all()}
          dim_params_fixer = models.AutoDimParams(default_by_name=dim_params_default, default=4)
          updated_onnx_model = ShapeInferenceStage.execute(report, dim_params_fixer=dim_params_fixer)
          _store(report=report, doc_id=doc_id, report_db=report_db)

      is_shape_inference_success = report.stages.shape_inference is not None and report.stages.shape_inference.status == "fixed"
      stats.shape_inference.update(report.stages.shape_inference.status)
      live.update(_make_panel(4, report.model_id))

      if until_stage == ShapeInferenceStage.__name__ or not is_shape_inference_success:
          continue

      #### Batch Dim Stage
      #rich.print("==========[bold]Starting Batching Dimension Stage[/bold]==========")
      #rich.print("Determine the batching dimension in the model inputs and outputs, which is required for labeling in the IFC mechanism.")
      if is_shape_inference_success and (report.stages.batch_dim is None or _force_recompute(report, BatchDimStage, recompute_cfg)):

          E = tinydb.Query()
          res = exception_db.search(E.model_id == report.model_id)
          assert len(res) <= 1, f"Duplicate exceptions for model {report.model_id}: {res}"

          # find dim param names that are used to indicate that this is a batch dim, e.g., `batch_size`
          Q = tinydb.Query()
          batch_dims_default: set[str] = {x["name"] for x in dimparam_db.search(Q.is_batch_dim == True)}

          if res: # have custom info for this model
            custom = res[0]
            BatchDimStage.execute(report,
                                  batching_status=custom["batch_status"],
                                  outputs_public=custom.get("outputs", {}).get("public"),
                                  outputs_batch_dim_custom=custom.get("outputs", {}).get("batch_dim"),
                                  inputs_public=custom.get("inputs", {}).get("public"),
                                  inputs_batch_dim_custom = custom.get("inputs", {}).get("batch_dim"),
                                  batch_dim_params_default=batch_dims_default,
                                  )
          else:
            BatchDimStage.execute(report,
                                  batching_status="unknown",
                                  outputs_public=set(),
                                  outputs_batch_dim_custom={},
                                  inputs_public=set(),
                                  inputs_batch_dim_custom = {},
                                  batch_dim_params_default=batch_dims_default,
                                  )
          _store(report=report, doc_id=doc_id, report_db=report_db)

      is_batchdim_success = report.stages.batch_dim is not None and report.stages.batch_dim.status == "batch"
      stats.batch_dim.update(report.stages.batch_dim.status)
      live.update(_make_panel(5, report.model_id))

      if until_stage == BatchDimStage.__name__ or not is_batchdim_success:
          continue

      #### Operator Stage
      #rich.print("==========[bold]Starting Operator Stage[/bold]==========")
      #rich.print("Check whether all model operators are implemented in the IFC mechanism.")
      # if report.stages.load is not None and report.stages.load.status == "success": # always recompute the operator availability
      if is_batchdim_success and (report.stages.operator is None or _force_recompute(report, OperatorStage, recompute_cfg)):
        OperatorStage.execute(report)
        _store(report=report, doc_id=doc_id, report_db=report_db)

      is_operator_success = report.stages.operator is not None and report.stages.operator.status == "available"
      stats.operator.update(report.stages.operator.status)
      live.update(_make_panel(6, report.model_id))

      if until_stage == OperatorStage.__name__ or not is_operator_success:
        continue

      #### Interference Check Stage
      #rich.print("==========[bold]Starting Interference Check Stage[/bold]==========")
      #rich.print("Perform the actual interference check to determine whether a model is batching safe.")
      if report.stages.batch_dim is not None and report.stages.batch_dim.status == "batch" and is_operator_success and (report.stages.interference_check is None or _force_recompute(report, InterferenceCheckStage, recompute_cfg)):
          if updated_onnx_model is None:
            # TODO: do code deduplication

            dim_params_default = {x["name"]: x["value"] for x in dimparam_db.all()}
            dim_params_fixer = models.AutoDimParams(default_by_name=dim_params_default, default=4)
            updated_onnx_model = ShapeInferenceStage.execute(report, dim_params_fixer=dim_params_fixer)
          InterferenceCheckStage.execute(report, updated_onnx_model=updated_onnx_model)

          #print(f"-----------> {report.stages.interference_check}")
          _store(report=report, doc_id=doc_id, report_db=report_db)

      is_interference_success = report.stages.interference_check is not None and report.stages.interference_check.status == "noninterference"
      stats.interference_check.update(report.stages.interference_check.status)
      live.update(_make_panel(1, report.model_id))

@attr.define
class Report:
  model_id: str
  stages: "StageInfo"
  model: models.Model | None = None


@attr.define
class SelectionStage:
  status: Literal["included", "skipped"]
  metadata: dict | None = None
  desc: str | None = None

  def check(self):
    if self.status == "included":
      assert self.metadata is not None, "Metadata none for included status"

  @classmethod
  def clear(cls, report: Report):
    report.stages.selection = None
    for Stage in [DownloadStage, LoadStage, ShapeInferenceStage, BatchDimStage, OperatorStage, InterferenceCheckStage]:
      Stage.clear(report)

  @classmethod
  def execute(cls, sources: List[str]) -> Tuple[List[Report], List[Dict]]:

    if len(sources) != 1 or sources[0] != "hf-onnx-community":
      raise ValueError(f"Not all sources {sources} are currently implementd.")

    #  {repo_id: {"models": ['model1.onnx', 'model2.onnx'], "files": [{"file": "model1.onnx", ...}]}}
    candidates = huggingface_onnx.get_candidates()

    # [{'repo_id': 'xx','files': [{'file': 'dequantizer.onnx', 'bytes': 181,}],'name': 'dequantizer'}, ..]
    download_plan = huggingface_onnx.select_models(candidates)

    reports = []
    n_included = 0

    selected = set()
    for x in download_plan:
      repo_id = x["repo_id"]
      for f in x["files"]:
        if f["file"].endswith(".onnx"):
            model_id = os.path.join(repo_id, f["file"])
            selected.add(model_id)

            report = Report(
                model_id=model_id,
                stages=StageInfo(selection=SelectionStage(status="included", metadata=x)),
                model = None
            )
            reports.append(report)

    for repo_id, x in candidates.items():
      for m in x["models"]:
        model_id = os.path.join(repo_id, m)
        if model_id in selected:
          n_included += 1
        else:
          # SelectionStage.clear(report)
          report = Report(
                model_id=model_id,
                stages=StageInfo(selection=SelectionStage(status="skipped", metadata=None)),
                model = None)
          reports.append(report)

    assert len(selected) == n_included, f"{len(selected)=} vs. {n_included}"


    dim_param_values: list[str] = huggingface_onnx.get_dim_params()

    return reports, dim_param_values

@attr.define
class DownloadStage:
  status: Literal["success", "skipped", "failed"]

  local_path: str | None = None

  def check(self):
    if self.status == "success":
      assert self.local_path is not None and os.path.isfile(self.local_path), f"Download success means file exists locally: {self.local_path}"

  @classmethod
  def clear(cls, report: Report):
    report.stages.download = None
    report.model = None
    for Stage in [LoadStage, ShapeInferenceStage, BatchDimStage, OperatorStage, InterferenceCheckStage]:
      Stage.clear(report)

  @classmethod
  def execute(cls, report: Report, local_dir: str) -> models.Model:

    os.makedirs(local_dir, exist_ok=True)

    metadata = report.stages.selection.metadata
    main_path = huggingface_onnx._download_model(metadata, local_dir)

    if main_path is None:
      DownloadStage.clear(report)
      report.stages.download = DownloadStage(status="failed", local_path=None)
    else:
      report.model = models.Model(
        name=report.model_id,
        src=models.LocalSrc(path=main_path),
        labeler=None,
        dim_params=None,
      )

      report.stages.download = DownloadStage(status="success", local_path=main_path)


  @classmethod
  def execute_all(cls, reports: List[Tuple[int, Report]], local_dir: str) -> List[int]:

    os.makedirs(local_dir, exist_ok=True)

    n_selected = 0
    n_successes = 0
    n_failed = 0

    to_download_bytes = 0
    to_download: list[Report] = list()
    updated_doc_ids = []

    for doc_id, report in reports:

        if report.stages.selection.status != "included":
          continue

        updated_doc_ids.append(doc_id)
        n_selected += 1

        model = report.stages.selection.metadata
        # {"repo_id": .., "name": .., "files": [{"file": .., }, ..]}
        main_file = None
        has_all_files = True
        for f in model["files"]:
            if f["file"].endswith(".onnx"):
                main_file =  f["file"]
            #if model["repo_id"] in local_cache and  f["file"] in local_cache[model["repo_id"]]
            if os.path.isfile(os.path.join(local_dir, model["repo_id"], f["file"])):
                # cache hit
                pass
            else:
                # cache miss
                has_all_files = False

        if has_all_files:
            assert(main_file is not None)
            local_path = os.path.join(local_dir, model["repo_id"], main_file)
            report.stages.download = DownloadStage(status="success", local_path=local_path)

            report.model = models.Model(
              name=report.model_id,
              src=models.LocalSrc(path=local_path),
              labeler=None,
              dim_params=None,
            )
        else:
            for f in model["files"]:
              to_download_bytes += f["bytes"]
            to_download.append(report)

    if len(to_download) == 0:
      return updated_doc_ids

    elif to_download_bytes > 500_000_000:
        prompt = (
            f"Do you want to download all {len(to_download)}/{n_selected} unavailable models?"
            f" Requires {huggingface_onnx._format_human_readable_size(to_download_bytes)} space."
        )

        if not _confirm(prompt):
          # mark all as skipped
          for report in to_download:
            report.stages.download = DownloadStage(status="skipped", local_path=None)
          return updated_doc_ids

    n_successes = 0
    n_failed = 0

    with progress.Progress() as prog:

        task1 = prog.add_task("[red]Download Models...", total=len(to_download))
        task2 = prog.add_task("[green]Download Total Size...", total=to_download_bytes)

        for report in to_download:
            DownloadStage.execute(report, local_dir)
            metadata=report.stages.selection.metadata

            is_success = report.stages.download.status == "success"
            if is_success:
              n_successes += 1
            else:
              n_failed += 1

            prog.update(task1, advance=1)
            prog.update(task2, advance=sum(f["bytes"] for f in metadata["files"]))

    print(
        f"Downloaded {n_successes} models, failed to download {n_failed} models"
    )

    return updated_doc_ids



domain = str
op = str
@attr.define
class LoadStage:
  status: Literal["success", "failed"]

  operator_histogram: dict[domain, dict[op, int]] | None = None
  dim_params: list[str] | None = None
  inputs: dict[str, tuple] | None = None
  outputs: dict[str, tuple] | None = None
  opset: list[dict[str, Any]] | None = None
  errors: list[str] | None = None

  def check(self):
    if self.status == "success":
      assert self.operator_histogram is not None
      assert self.dim_params is not None
      assert self.inputs is not None
      assert self.outputs is not None
      assert self.opset is not None
      assert self.errors is None
    else:
      assert self.status == "failed"
      assert len(self.errors) > 0

  @classmethod
  def clear(cls, report: Report):
    if report.model is not None:
      report.model.clear_model()
    report.stages.load = None
    for Stage in [ShapeInferenceStage, BatchDimStage, OperatorStage, InterferenceCheckStage]:
      Stage.clear(report)



  @classmethod
  def execute(cls, report: Report):
    try:
        print("Load + check model...")
        onnx_model = report.model.load(check=True)

        check_graph(onnx_model)
    except Exception as e:
        print(f"Error loading model: {e}")
        report.stages.load = LoadStage(status="failed", errors=str(e).splitlines())
        return

    operators = model_summary.model_operators(onnx_model)
    opset = model_summary.get_opset(onnx_model)
    opset = [{"domain": o.domain, "version": o.version} for o in opset]
    dim_params, inputs, outputs = model_summary.model_inputs_outputs(onnx_model)
    report.stages.load = LoadStage(status="success", operator_histogram=operators, dim_params=list(dim_params), inputs=inputs, outputs=outputs, opset=opset, errors=None)




@attr.define
class ShapeInferenceStage:
  status: Literal["fixed", "dynamic", "failed"]

  dim_param_values: dict[str, int] | None = None # the chosen values for the dim_params of the model

  dynamic_output_nodes: List[dict[str, Any]] | None = None # context for dynamic

  errors: List[str] | None = None # context for failed

  def check(self):
    if self.status == "fixed":
      assert self.dynamic_output_nodes is None
      assert self.errors is None
    elif self.status == "dynamic":
      assert len(self.dynamic_output_nodes) > 0
      assert self.errors is None
    else:
      assert self.status == "failed"
      assert len(self.errors) > 0


  @classmethod
  def clear(cls, report: Report):
    report.stages.shape_inference = None
    for Stage in [BatchDimStage, OperatorStage, InterferenceCheckStage]:
      Stage.clear(report)

  @classmethod
  def execute(cls, report: Report, dim_params_fixer: models.AutoDimParams = None, tmp_model_path: str = "/tmp/model.onnx"):
    onnx_model = report.model.load()

    if dim_params_fixer is None:
      dim_params_fixer = models.AutoDimParams(default_by_name={}, default=4)
    dim_param_values = dim_params_fixer.assign_values(report.stages.load.dim_params)

    try:
      onnx_model_updated = fix_dim_params(onnx_model, dim_param_values=dim_param_values)
    except Exception as e:
      report.stages.shape_inference = ShapeInferenceStage(status="failed", dim_param_values=dim_param_values, dynamic_output_nodes=None, errors=["$FIX_DIM_PARAMS$"] + str(e).splitlines())
      return None

    report.model.clear_model() # free memory

    tmp_model_data_name = f"{os.path.basename(tmp_model_path)}.data"
    tmp_model_data_path = (f"{os.path.dirname(tmp_model_path)}/{tmp_model_data_name}")
    os.makedirs(os.path.dirname(tmp_model_path), exist_ok=True)

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
      # shape inference generates garbage files in cwd -> delete them
      os.chdir(tmp_dir)
      try:
          onnx_model_updated = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(onnx_model_updated)
      except Exception as e:
          pass # do nothing
      check_graph(onnx_model_updated)
      os.chdir(original_cwd)


    if os.path.isfile(tmp_model_path):
      os.remove(tmp_model_path)

    if os.path.isfile(tmp_model_data_path):
      os.remove(tmp_model_data_path)

    onnx.save(
      onnx_model_updated,
      tmp_model_path,
      size_threshold=0,
      save_as_external_data=True,
      location=tmp_model_data_name
    )

    # after saving model needs to be reloaded (old onnx_model_updated cannot be used anymore)
    onnx_model_updated = onnx.load(tmp_model_path)

    try:
      onnx.checker.check_model(tmp_model_path, check_custom_domain=False)
      check_graph(onnx_model_updated)

    except Exception as e:
      report.stages.shape_inference = ShapeInferenceStage(status="failed", dim_param_values=dim_param_values, dynamic_output_nodes=None, errors=["$CHECK$"] + str(e).splitlines())
      return None

    is_fixed, non_fixed_output_nodes = is_fixed_model(onnx_model_updated)


    if is_fixed:
      status = "fixed"
      assert non_fixed_output_nodes is None or len(non_fixed_output_nodes) == 0
      non_fixed_output_nodes = None
    else:
      status = "dynamic"

    report.stages.shape_inference = ShapeInferenceStage(status=status, dim_param_values=dim_param_values, dynamic_output_nodes=non_fixed_output_nodes, errors=None)

    return onnx_model_updated


@attr.define
class BatchDimStage:
  status: Literal["batch", "unknown", "nobatch"]

  batch_dim_params: list[str] | None = None

  inputs_public: list[str] | None = None
  outputs_public: list[str] | None = None

  inputs_batch_dim: dict[str, int] | None = None
  outputs_batch_dim: dict[str, int] | None = None

  inputs_missing_batch_dim: list[str] | None = None
  outputs_missing_batch_dim: list[str] | None = None

  def check(self):
    if self.status == "batch":
      assert self.inputs_batch_dim is not None
      assert self.outputs_batch_dim is not None
      assert self.inputs_public is not None
      assert self.outputs_public is not None

  @classmethod
  def clear(cls, report: Report):
    report.stages.batch_dim = None
#    if report.model is not None:
#      report.model.labeler = None
    for Stage in [OperatorStage, InterferenceCheckStage]:
      Stage.clear(report)


  @classmethod
  def execute(cls, report: Report, batching_status: Literal["nobatch", "unknown", "batch"], inputs_public: set[str], outputs_public: set[str], inputs_batch_dim_custom: dict[str, int], outputs_batch_dim_custom: dict[str, int], batch_dim_params_default: set[str]):

    if batching_status == "nobatch":
      report.stages.batch_dim = BatchDimStage(status="nobatch")
      return

    if inputs_public or outputs_public:
      raise ValueError("Public Inputs and/or Outputs are Not Supported Yet: Missing option for labeling the tensors accordingly.")

    ld = report.stages.load

    def _extract_batch_dims(tensors: dict[str, tuple], custom: dict[str, int]) -> Tuple[dict[str, int], set[str], set[str]]:
        batch_dims = {}

        batch_dim_params = set()

        for name, shape in tensors.items():
            for i, dim in enumerate(shape):
                if dim in batch_dim_params_default:
                    assert name not in batch_dims, f"Duplicate occurence of batch dims: {name} {shape}"
                    assert dim in ld.dim_params
                    batch_dim_params.add(dim)
                    batch_dims[name] = i

        for name, custom_batch_dim in custom.items():
          if name in batch_dims:
            print(f"For model {report.model_id}.{name} ({tensors[name]}) override default batch dim ({batch_dims[name]}) -> custom ({custom_batch_dim})")
          batch_dims[name] = custom_batch_dim

        missing = set(tensors.keys()).difference(batch_dims.keys())
        return batch_dims, batch_dim_params, missing

    inputs_batch_dim, inputs_used_defaults, inputs_missing_batch_dim = _extract_batch_dims({k: v for k, v in ld.inputs.items() if k not in inputs_public}, custom=inputs_batch_dim_custom)
    outputs_batch_dim, outputs_used_defaults, outputs_missing_batch_dim = _extract_batch_dims({k: v for k, v in ld.outputs.items() if k not in outputs_public}, custom=outputs_batch_dim_custom)
    if inputs_missing_batch_dim or outputs_missing_batch_dim:
      status = "unknown"
    #elif inputs_public or outputs_public or inputs_batch_dim_custom or outputs_batch_dim_custom:
    #  status = "manual"
    else:
      status = "batch"

    batch_dim_params = inputs_used_defaults.union(outputs_used_defaults)

    if len(batch_dim_params) > 1:
      warnings.warn(f"Model {report.model_id} has more than one name for batching dimension: {batch_dim_params}")

    report.stages.batch_dim = BatchDimStage(status=status,
                                              batch_dim_params=list(batch_dim_params),
                                              inputs_public=list(inputs_public),
                                              outputs_public=list(outputs_public),
                                              inputs_batch_dim=inputs_batch_dim,
                                              outputs_batch_dim=outputs_batch_dim,
                                              inputs_missing_batch_dim=list(inputs_missing_batch_dim) if inputs_missing_batch_dim else None,
                                              outputs_missing_batch_dim=list(outputs_missing_batch_dim) if outputs_missing_batch_dim else None)



@attr.define
class OperatorStage:
  status: Literal["available", "missing"]

  missing_taint_ops: dict[domain, List[op]]
  missing_data_ops: dict[domain, List[op]]

  def check(self):
    if self.status == "available":
      assert sum(len(v) for _, v in self.missing_data_ops.items()) == 0
      assert sum(len(v) for _, v in self.missing_taint_ops.items()) == 0

  @classmethod
  def clear(cls, report: Report):
    report.stages.operator = None
    for Stage in [InterferenceCheckStage]:
      Stage.clear(report)

  @classmethod
  def execute(cls, report: Report):
    operators_needed = report.stages.load.operator_histogram
    opset = report.stages.load.opset

    _, missing_taint_ops, missing_data_ops = model_summary.missing_operators_inner(operators_needed, opset)

    n_missing_taint_ops = sum(len(v) for _, v in missing_taint_ops.items())
    n_missing_data_ops = sum(len(v) for _, v in missing_data_ops.items())

    if n_missing_data_ops + n_missing_taint_ops > 0:
      status = "missing"
    else:
      status = "available"

    report.stages.operator = OperatorStage(status=status, missing_taint_ops=missing_taint_ops, missing_data_ops=missing_data_ops)


@attr.define
class InterferenceCheckStage:
  status: Literal["noninterference", "interference", "failed"]

  runtime_sec: dict[str, int]

  proof_report: report.ProofReport | None = None

  errors: List[str] | None = None # context for failed

  def check(self):
    if self.status in ["noninterference", "interference"]:
      assert self.errors is None
      assert self.proof_report is not None
    else:
      assert self.status == "failed"
      assert len(self.errors) > 0

  @classmethod
  def clear(cls, report: Report):
    report.stages.interference_check = None

  @classmethod
  def execute(cls, report: Report, updated_onnx_model: onnx.ModelProto):

    runtime_sec = {}

    start = time.time()

    if updated_onnx_model is None:
      report.stages.interference_check = InterferenceCheckStage(status="failed", runtime_sec=0, proof_report=None, errors=["No fixed size model received."])
      return

    if report.model.labeler is None:
      combined_batch_dim = {}
      for k, v in report.stages.batch_dim.inputs_batch_dim.items():
        combined_batch_dim[k] = v
      for k, v in report.stages.batch_dim.outputs_batch_dim.items():
        assert k not in combined_batch_dim, f"Duplicate input output name: {k}  {report.model_id}"
        combined_batch_dim[k] = v
      report.model.labeler = labels.BatchDimLabeler(io_batch_dim=0, io_batch_dim_exceptions=combined_batch_dim)

    # perform the non-interference check
    try:



      mid = time.time()
      colored_inputs, expected_colored_outputs = (
        # TODO: I don't need to rely on the "model" for this logic
          report.model.labeler.get_labeled_inputs_outputs(updated_onnx_model, {})
      )
      skip_ids = ["whisper-large-v3-", "whisper-d-v1a", "kb-whisper-large-ONNX", "kotoba-whisper-v2.2"]
      if  any(x in report.model_id for x in skip_ids):
        raise ValueError(f"skipping model: {report.model_id}")

      config_class.config.update(
        "jaxort_only_allow_initializers_as_static_args", False
      )

      jax.config.update("jax_enable_x64", True)
      taint_config.config.update("taint_dtype", "uint16")


      backend = jax_taint_backend.TaintBackendRep(updated_onnx_model)
      mid = time.time()
      runtime_sec["setup"] = mid - start

      proof_report = backend.run(colored_inputs, expected_colored_outputs, info_level=0)
    except Exception as e:
      runtime_sec["run"] = time.time() - mid
      report.stages.interference_check = InterferenceCheckStage(status="failed", runtime_sec=runtime_sec, proof_report=None, errors=str(e).splitlines())
      print(f"\n\n??????IFC ERROR: {str(e)}\n\n")
      return
    runtime_sec["run"] = time.time() - mid

    if proof_report.proof_status == "PASSED":
      status = "noninterference"
      print(f"\n\n??????IFC REPORT: {proof_report}\n\n")
    elif proof_report.proof_status == "FAILED":
      status = "interference"
      print(f"\n\n??????IFC REPORT: -> Found Interference \n\n")
    else:
      raise ValueError(f"Unknown proof status: {proof_report.proof_status}")

    report.stages.interference_check = InterferenceCheckStage(status=status, runtime_sec=runtime_sec, proof_report=proof_report, errors=None)


def check_graph(onnx_model: onnx.ModelProto):
  graph = onnx_model.graph
  graph = onnx_utils.sanitize_tensor_names_in_graph(graph)
  onnx_graph.OnnxGraph(graph)

def get_report_converter():
  def structure_union_int_str(val, _):
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        return val
    raise ValueError(f"Cannot structure {val!r} as Union[int, str]")

  converter = cattrs.Converter()
  converter.register_structure_hook(Union[int, str], structure_union_int_str)

  return converter


def _confirm(prompt_str) -> bool:
  choice = prompt.Prompt.ask(prompt_str, choices=["y", "n"], default="y")
  return choice == "y"


def is_field_included(attr, value):
    # true -> included
    return value is not None and not attr.metadata.get("exclude", False)

def _update_report_db(reports: List[Report], report_db, mode: Literal["ReplaceAll", "Upsert", "AddIfNotExists"] = "ReplaceAll"):

    n_reports = len(reports)

    if mode == "ReplaceAll":
      n_current = len(report_db)
      if n_current > 0 and not _confirm(f"Replace all {n_current} reports with the new {len(reports)} reports?"):
        raise ValueError("-> aborted")
      report_db.truncate() # delete all
      for report in reports:
        report_db.insert(attr.asdict(report, filter=is_field_included))

    elif mode == "Upsert" or mode == "AddIfNotExists":
      R = tinydb.Query()
      n_updates = 0
      non_existing_reports = []
      for report in reports:
        c = report_db.count(R.model_id == report.model_id)
        if c == 0:
          non_existing_reports.append(report)
        assert c <= 1
        n_updates += c
      n_insert = len(non_existing_reports)

      if mode == "Upsert":
        if not _confirm(f"Replace {n_updates} reports and Insert {n_insert} new reports?"):
            raise ValueError("-> aborted")
        for report in reports:
          report_db.upsert(attr.asdict(report, filter=is_field_included), R.model_id == report.model_id)
      elif mode == "AddIfNotExists":
        if not _confirm(f"Add {n_insert} new reports to the existing {n_reports}? ({n_updates} duplicates are skipped)."):
            raise ValueError("-> aborted")
        for report in non_existing_reports:
          report_db.insert(attr.asdict(report, filter=is_field_included))
      else:
        raise ValueError("Unreachable")
    else:
      raise ValueError(f"unknown {mode}")


def _store(report: Report, doc_id: int, report_db: tinydb.table):
  report.stages.check()
  report_db.update(attr.asdict(report, filter=is_field_included), doc_ids=[doc_id])


def fix_dim_params(
    onnx_model: onnx.ModelProto, dim_param_values: Dict[str, int]
) -> onnx.ModelProto:
  """Replaces dim_params in the model with the given values."""

  _, inputs, outputs = model_summary.model_inputs_outputs(onnx_model)

  def _get_dim_params(info) -> set[str]:
    detected = set()
    for _name, shape in info.items():
      for s in shape:
        if isinstance(s, str):
          detected.add(s)
    return detected


  def _replace_dim_params(shape: Sequence[Union[int, str]], replace_only: Optional[set[str]]=None):
    new_shape = []
    for s in shape:
      if isinstance(s, str):
        if replace_only is None or s in replace_only:
          new_shape.append(dim_param_values[s])
      elif isinstance(s, int):
        new_shape.append(s)
      else:
        raise ValueError(f"Unknown dim type: {s}")
    return new_shape


  dim_params_in_input: set[str] = _get_dim_params(inputs)

  inputs_fixed = {
      name: _replace_dim_params(shape) for name, shape in inputs.items()
  }

  # in outputs we only want to replace the dim params also present in the inputs, the shape inference must be able to set them by itself
  outputs_fixed = {
      name: _replace_dim_params(shape, replace_only=dim_params_in_input) for name, shape in outputs.items()
  }

  updated_model = _update_inputs_outputs_dims(
      model=onnx_model, input_dims=inputs_fixed, output_dims=outputs_fixed
  )

  return updated_model



def is_fixed_model(
    onnx_model: onnx.ModelProto,
) -> Tuple[bool, list[dict[str, Any]]]:
  """Checks if the model only contains nodes with fixed output shapes."""

  # onnx stores tensor shape information in the value_info field.
  fixed_tensors = set()
  unknown_tensor_types = dict()
  for info in onnx_model.graph.value_info:
    if info.type is not None:
      value_type = info.type.WhichOneof("value")
      if value_type == "tensor_type":
        shape = info.type.tensor_type.shape
        if all(x.WhichOneof("value") == "dim_value" for x in shape.dim):
          fixed_tensors.add(info.name)
      else:
        warnings.warn(f"Unknown value type: {value_type} {info.name}")
        if value_type not in unknown_tensor_types:
          unknown_tensor_types[value_type] = 1
        unknown_tensor_types[value_type] += 1

  if unknown_tensor_types:
    warnings.warn(
        f"Unknown tensor types: {unknown_tensor_types}. I those can also"
        " considered to be fixed, then this model may be wrongly excluded."
    )

  non_fixed_output_nodes = list()
  for node in onnx_model.graph.node:
    has_fixed_output = all(x in fixed_tensors or not x for x in node.output)
    if not has_fixed_output:
      location = {
          "domain": node.domain,
          "op_type": node.op_type,
          "name": node.name,
          "outputs": [x for x in node.output if x and x not in fixed_tensors],
      }
      non_fixed_output_nodes.append(location)

  is_fixed = not non_fixed_output_nodes

  return is_fixed, non_fixed_output_nodes


def _update_inputs_outputs_dims(
    model: onnx.ModelProto,
    input_dims: Dict[str, List[Any]],
    output_dims: Dict[str, List[Any]],
) -> onnx.ModelProto:
  """Updates input and output dimensions of the model with the given values."""

  # NOTE: This is a copy of onnx.tools.update_model_dims.update_model_dims()
  #        but with the checker removed because this will fail for large models.

  dim_param_set: Set[str] = set()

  def init_dim_param_set(
      dim_param_set: Set[str], value_infos: List[onnx.ValueInfoProto]
  ) -> None:
    for info in value_infos:
      shape = info.type.tensor_type.shape
      for dim in shape.dim:
        if dim.HasField("dim_param"):
          dim_param_set.add(dim.dim_param)  # type: ignore

  init_dim_param_set(dim_param_set, model.graph.input)  # type: ignore
  init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore
  init_dim_param_set(dim_param_set, model.graph.value_info)  # type: ignore

  def update_dim(
      tensor: onnx.ValueInfoProto, dim: Any, j: int, name: str
  ) -> None:
    dim_proto = tensor.type.tensor_type.shape.dim[j]
    if isinstance(dim, int):
      if dim >= 0:
        if dim_proto.HasField("dim_value") and dim_proto.dim_value != dim:
          raise ValueError(
              "Unable to set dimension value to {} for axis {} of {}."
              " Contradicts existing dimension value {}.".format(
                  dim, j, name, dim_proto.dim_value
              )
          )
        dim_proto.dim_value = dim
      else:
        generated_dim_param = name + "_" + str(j)
        if generated_dim_param in dim_param_set:
          raise ValueError(
              "Unable to generate unique dim_param for axis {} of {}. Please"
              " manually provide a dim_param value.".format(j, name)
          )
        dim_proto.dim_param = generated_dim_param
    elif isinstance(dim, str):
      dim_proto.dim_param = dim
    else:
      raise ValueError(
          "Only int or str is accepted as dimension value, incorrect type:"
          f" {type(dim)}"
      )

  for input_proto in model.graph.input:
    input_name = input_proto.name
    input_dim_arr = input_dims[input_name]
    for j, dim in enumerate(input_dim_arr):
      update_dim(input_proto, dim, j, input_name)

  for output_proto in model.graph.output:
    output_name = output_proto.name
    output_dim_arr = output_dims[output_name]
    for j, dim in enumerate(output_dim_arr):
      update_dim(output_proto, dim, j, output_name)

  # onnx.checker.check_model(model)
  return model




@attr.define
class StageInfo:
  selection: SelectionStage | None = None
  download: DownloadStage | None = None
  load: LoadStage | None = None
  shape_inference: ShapeInferenceStage | None = None
  batch_dim: BatchDimStage | None = None
  operator: OperatorStage | None = None
  interference_check: InterferenceCheckStage | None = None

  def get_stages(self) -> List[Any]:
    return [self.selection, self.download, self.load, self.shape_inference, self.batch_dim, self.operator, self.interference_check]


  def check(self):
    stages = [self.selection, self.download, self.load, self.shape_inference, self.batch_dim, self.operator, self.interference_check]

    first_none_stage = None
    for i, stage in enumerate(stages):
      if stage is not None:
        stage.check()

        if first_none_stage is not None:
          raise ValueError(f"Stage dependency violation: Previous stage is None: {first_none_stage}, and later stage is not None: {i} ({stage})")
      else:
        if first_none_stage is None:
          first_none_stage = i


#  def is_candidate(self) -> bool:
#    """The model could be supported (i.e., cannot delete yet)"""
#
#    is_c = self.stages.selection is None or self.stages.selection.status == "included"
#    #is_c = is_c and self.stages.download is None or self.stages.download.status
#    is_c = is_c and self.stages.load is None or self.stages.load.status == "success"
#    is_c = is_c and self.stages.shape_inference is None or self.stages.shape_inference.status == "fixed"
#    is_c = is_c and self.stages.batch_dim is None or self.stages.batch_dim.status in ["batch", "unknown"]
#
#    return is_c




#def main(argv: Sequence[str]) -> None:
#  del argv
#
#  if not FLAGS.skip:
#
#    if FLAGS.hub == "huggingface":
#      models_lst = huggingface_onnx.get_models(FLAGS.models, repo_subdir="")
#    elif FLAGS.hub == "github":
#      models_lst = github_onnx.get_models(FLAGS.models, repo_subdir="")
#    else:
#      raise ValueError(f"Unsupported hub: {FLAGS.hub}")
#
#    check_requirements(
#        models_lst,
#        output_json=FLAGS.output,
#        output_tmp=FLAGS.state,
#        amend_tmp=FLAGS.amend,
#    )
#
#  report_requirements.report_cli(reports_json=FLAGS.output)
#
#
#if __name__ == "__main__":
#  app.run(main)