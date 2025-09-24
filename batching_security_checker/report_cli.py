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

from absl import app, flags

from batching_security_checker.report import customization
from batching_security_checker.report import stages


FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "mode",
    "report",
    ["report", "batchio"],
    "Mode to run: either 'report' or 'batchio'.",
)

flags.DEFINE_string(
    "database",
    None,
    "Path to the report tinydb json.",
    short_name="d",
    required=True
)

flags.DEFINE_string(
    "models",
    None,
    "Directory where models are stored. (only required in 'report' mode)",
    short_name="m"
)

flags.DEFINE_enum(
    "until_stage",
    None,
    [
        "SelectionStage",
        "DownloadStage",
        "LoadStage",
        "ShapeInferenceStage",
        "BatchDimStage",
        "OperatorStage",
        "InterferenceCheckStage"
    ],
    "Optional: Stop after a given stage. Only applies in 'reports' mode."
)



def _force(report: stages.Report) -> bool:

    st = report.stages

    if st.load is not None and st.load.status in ["failed"]:
        error_reasons = ["No Op registered for SimplifiedLayerNormalization with domain_version of "]
        for search in error_reasons:
            if any(search in e for e in st.load.errors):
                return True

    if st.shape_inference is not None and st.shape_inference.status in ["failed"]:

        error_reasons = []
        for search in error_reasons:
            if any(search in e for e in st.shape_inference.errors):
                return True

    if st.interference_check is not None and st.interference_check.status in ["failed"]:

        error_reasons = ["Only one live display may be active",
                        "combined_args must be even length: len=9",
                         "can only concatenate tuple (not ",
                         "onnx_taint_clip() got an unexpected keyword argument",
                         "only version with two args is currently supported: len(input_args)=5",
                         "Erf version 9 is not implemented. Only have those versions:",
                         "not found in graph: you may need to run shape inference on the model first"]
        for search in error_reasons:
            if any(search in e for e in st.interference_check.errors):
                return True

    return False

def main(argv):
    if FLAGS.mode == "batchio":
        customization.interactive_batch_dim_cli(FLAGS.database)
    elif FLAGS.mode == "report":
        if FLAGS.models is None:
            raise ValueError("You must provide --models for 'report' mode.")

        # NOTE: if a previous stage is recomputed, the consecutive stages are usually also recomputed
        recompute_cfg = {
            "SelectionStage": False,
            "DownloadStage": False,
            "LoadStage": _force,
            "ShapeInferenceStage": _force,
            "BatchDimStage": _force,
            "OperatorStage": _force,
            "InterferenceCheckStage":  _force,
        }

        stages.create_reports(
            database_path=FLAGS.database,
            local_model_dir=FLAGS.models,
            recompute_cfg=recompute_cfg,
            until_stage=FLAGS.until_stage
        )


if __name__ == "__main__":
    app.run(main)