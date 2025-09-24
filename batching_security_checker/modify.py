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

"""Modify an ONNX model to introduce an intra-batch leakage."""

from collections.abc import Sequence
import os
import shutil

from absl import app
from absl import flags
import onnx

from batching_security_checker.core import util
from batching_security_checker.leaks import subgraphs


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dir", help="Model directory", required=True
)
flags.DEFINE_string("model", None, "Model name (without .onnx)", required=True)
flags.DEFINE_string("leak", "reroute", "Leak name")
flags.DEFINE_string(
    "loc", None, "Location (tensor name) of leak in model", required=True
)


#   --model=gemma-2b-it-fp16-onnx --loc=logits
#   --model=mnist_noleak --loc=/Reshape_output_0
#   --model=gemma-2b-it-fp16-onnx --loc="present.0.key" --leak=trigger
def main(argv: Sequence[str]) -> None:
  del argv

  model_path = util.resolve_model_path(FLAGS.dir, FLAGS.model)
  print(f"Loading model from {model_path}")
  model = onnx.load(model_path)

  if FLAGS.leak == "reroute":
    subgraphs.RerouteSubgraph(model=model, name=FLAGS.loc)
  elif FLAGS.leak == "trigger":
    subgraphs.TokenTriggerSubgraph(model=model, name=FLAGS.loc)
  else:
    raise ValueError(f"Unknown leak: {FLAGS.leak}")

  model_out_path = model_path.replace(
      FLAGS.model, f"{FLAGS.model}_{FLAGS.leak}"
  )
  model_out_dir = os.path.dirname(model_out_path)
  os.makedirs(model_out_dir, exist_ok=True)

  print(f"Saving model to {model_out_path}")
  onnx.save_model(
      model,
      model_out_path,
      save_as_external_data=True,
      location=f"{os.path.basename(model_out_path)}.data",
  )

  tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer.model")
  if os.path.isfile(tokenizer_path):
    print(f"Copying tokenizer to {model_out_dir}")
    shutil.copy2(tokenizer_path, os.path.join(model_out_dir, "tokenizer.model"))

  onnx.shape_inference.infer_shapes_path(model_out_path, model_out_path)
  onnx.checker.check_model(model_out_path, full_check=True, check_custom_domain=False)


if __name__ == "__main__":
  app.run(main)
