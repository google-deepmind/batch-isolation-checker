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

"""Script for running LLM inference."""

import glob
import os
import time
from typing import Optional, Sequence

from absl import app
from absl import flags
import gemma
from rich import columns as rich_columns
from rich import console as rich_console
from rich import table as rich_table

FLAGS = flags.FLAGS
flags.DEFINE_string("dir", None, "Model directory")
flags.DEFINE_string("model", None, "Model name (without .onnx)", required=True)


flags.DEFINE_string("p0", None, "Prompt 0")
flags.DEFINE_string("p1", None, "Prompt 1")
flags.DEFINE_string("p2", None, "Prompt 2")
flags.DEFINE_string("p3", None, "Prompt 3")

scenarios = {
    "manual": [],
    "no": [
        "Story: A bank is",
        "The meaning of life is",
    ],
    "get": [
        "@@get others result:",
        "The meaning of life is",
    ],
    "set": [
        "@@set A bank is",
        "The meaning of life is",
    ],
}

flags.DEFINE_string(
    "mode",
    "manual",
    f"Scenario to run: {list(scenarios.keys())} ",
)


def run_gemma():
  """Run the Gemma model."""

  model_dir = os.path.join(FLAGS.dir, FLAGS.model)

  models = glob.glob(os.path.join(model_dir, "*.onnx"))
  if len(models) != 1:
    raise ValueError(f"Expected one model in {model_dir}, but found {models}")
  model_path = models[0]

  model = gemma.GemmaForCausalLM(model_path)

  # Generate the response.
  start = time.time()

  if FLAGS.mode not in scenarios:
    raise ValueError(f"Unknown mode: {FLAGS.mode}  {list(scenarios.keys())}")
  elif FLAGS.mode == "manual":
    prompts = [
        p for p in [FLAGS.p0, FLAGS.p1, FLAGS.p2, FLAGS.p3] if p is not None
    ]
    if not prompts:
      raise ValueError(
          f"No prompts for mode: {FLAGS.mode}, specify with --p0='..' --p1='..'"
          " --p2='..' --p3='..'"
      )
  else:
    prompts = scenarios[FLAGS.mode]

  print_output(prompts, None)
  results = model.generate(prompts)
  print(f"Generation took {int(time.time() - start)} seconds")
  print_output(prompts, results)


def print_output(prompts: Sequence[str], results: Optional[Sequence[str]]):
  """Print the generated results."""

  def create_table(user_id: str, prompt: str, result: Optional[str]):
    table = rich_table.Table(
        title=f"[bold magenta]User {user_id}[/bold magenta]", show_header=False
    )
    table.add_column(max_width=60)
    table.add_row(f"[bold]Prompt:[/bold] {prompt}")
    if result is not None:
      table.add_row(f"[bold]Result:[/bold] {result}")
    return table

  if results is None:
    results = [None] * len(prompts)

  tables = []

  for i, (prompt, result) in enumerate(zip(prompts, results, strict=True)):
    tables.append(create_table(i, prompt, result))
  columns = rich_columns.Columns(tables, equal=True)
  console = rich_console.Console()
  console.print(columns)


def main(argv: Sequence[str]) -> None:
  del argv
  run_gemma()


if __name__ == "__main__":
  app.run(main)
