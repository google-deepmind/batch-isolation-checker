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

import tinydb

from rich import prompt
from rich import console as rich_console
from rich import progress
import cattrs

from batching_security_checker.report.stages import Report, get_report_converter


def interactive_batch_dim_cli(database_path):
    """Go through the reports stored in the database, and for each report with an `unknown` batching stage, record the custom information."""

    db = tinydb.TinyDB(database_path)
    report_db = db.table("report")

    exception_db = db.table("exception")

    converter = get_report_converter()
    reports = report_db.all()


    console = rich_console.Console()

    n_updates = 0

    for report in reports: #, description="Processing Models...", total=len(reports)):
        doc_id = report.doc_id
        report = converter.structure(report, Report)
        stage = report.stages.batch_dim





        if stage is None or stage.status != "unknown":
            continue

        console.print(f"[bold blue]Model {report.model_id}[/bold blue]")
        console.print(f"[bold]Doc ID:[/bold] {doc_id}")
        console.print(f"[bold]Inputs:[/bold] {report.stages.load.inputs}")
        console.print(f"[bold]Outputs:[/bold] {report.stages.load.outputs}")

        E = tinydb.Query()
        cur_exception = exception_db.get(E.model_id == report.model_id)

        if cur_exception is not None:
            is_ok = prompt.Prompt.ask(f"Confirm existing exception: [bold]{cur_exception['batch_status']}[/bold]", console=console, choices=["y", "n"], default="y")
            if is_ok:
                continue


        model_level = prompt.Prompt.ask("Is this model batched?", console=console, choices=["batch", "nobatch"], default="nobatch")
        if model_level == "nobatch":
            stage.status = "nobatch"
            _mark_nobatch(model_id=report.model_id, exception_db=exception_db)
            console.print(f"[bold blue]-> Marked model {report.model_id} as nobatch[/bold blue]")
            n_updates += 1
            continue

        assert model_level == "batch"

        stage.inputs_batch_dim = {}
        stage.outputs_batch_dim = {}

        def handle_tensor_dims(tensors, kind, existing_dims):

            custom_batch_dims = {}
            public_io_tensors = set()
            for name, shape in tensors.items():
                cur_dim = existing_dims.get(name)
                shape_str = ", ".join(f"[{x}]" if i == cur_dim else str(x) for i, x in enumerate(shape))
                console.print(f"{kind} tensor [bold]{name}[/bold] shape: ({shape_str})")
                default_str = f"{cur_dim}" if cur_dim is not None else "none"
                response = prompt.Prompt.ask(
                    f"Batch dimension for {name}? (default: {default_str})",
                    console=console,
                    choices=[str(i) for i in range(len(shape))] + ["public", "skip"],
                    default=cur_dim if cur_dim is not None else "skip"
                )

                if response.isdigit():
                    custom_batch_dims[name] = int(response)
                    existing_dims[name] = int(response)
                elif response == "public":
                    if name in existing_dims:
                        del existing_dims[name]
                    public_io_tensors.add(name)
                elif response == "skip":
                    return None, None

            return custom_batch_dims, public_io_tensors

        custom_batch_dims_inputs, public_inputs = handle_tensor_dims(report.stages.load.inputs, "Input", stage.inputs_batch_dim)
        custom_batch_dims_outputs, public_outputs = handle_tensor_dims(report.stages.load.outputs, "Output", stage.outputs_batch_dim)

        if custom_batch_dims_inputs is None or custom_batch_dims_outputs is None:
            continue # skip report without storing update
        elif custom_batch_dims_inputs or public_inputs or custom_batch_dims_outputs or public_outputs:
            _mark_exception(model_id=report.model_id, custom_batch_dims_inputs=custom_batch_dims_inputs, custom_batch_dims_outputs=custom_batch_dims_outputs, public_inputs=list(public_inputs), public_outputs=list(public_outputs), exception_db=exception_db)
            console.print(f"[bold blue]-> Custom batching dimensions recorded for model {report.model_id}[/bold blue]")
            n_updates += 1

    console.print("==================================")
    console.print(f"Recorded {n_updates} custom batching logic for batching inputs and outputs.")
    console.print(f"Run the report check including the batching check for these exceptions to be used in  the reports.")
    console.print("==================================")


# exception_db schema:
# {"model_id" xxx , "inputs": {"batch_dim": {"name": idx}, "public": []}, "outputs": {"batch_dim": {"name": idx}, "public": []}}

def _mark_nobatch(model_id: int, exception_db):

    # record an exception
    E = tinydb.Query()
    doc_ids = exception_db.upsert({"model_id": model_id, "batch_status": "nobatch"}, E.model_id == model_id)
    assert len(doc_ids) == 1, f"Exception DB contains duplicate entries for {model_id=}: {doc_ids=}"


def _mark_exception(model_id: str, custom_batch_dims_inputs: dict[str,int], custom_batch_dims_outputs: dict[str,int], public_inputs: list[str], public_outputs: list[str], exception_db):
    d = {
        "model_id": model_id,
        "batch_status": "batch",
        "inputs": {"batch_dim": custom_batch_dims_inputs, "public":  public_inputs},
        "outputs": {"batch_dim": custom_batch_dims_outputs, "public": public_outputs}
        }

    # record an exception
    E = tinydb.Query()
    doc_ids = exception_db.upsert(d, E.model_id == model_id)
    assert len(doc_ids) == 1, f"Exception DB contains duplicate entries for {model_id=}: {doc_ids=}"



if __name__ == "__main__":
    database_path = "/home/ubuntu/batching-security-checker/data/report_db.json"
    interactive_batch_dim_cli(database_path)