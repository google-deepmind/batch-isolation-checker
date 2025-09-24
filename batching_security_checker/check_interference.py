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

"""Run the batching security checker on a model."""

from collections.abc import Sequence
import logging
import time

from absl import app
from absl import flags
import jax
from jaxonnxruntime import config_class
import rich

from batching_security_checker import backend as jax_taint_backend
from batching_security_checker.core import config_class as taint_config
from batching_security_checker.core import labels
from batching_security_checker.core import models
from batching_security_checker.core import util


logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
flags.DEFINE_string("family", "gemma", "Model family")

flags.DEFINE_string("dir", None, "Model directory")

flags.DEFINE_string("model", None, "Model name (without .onnx)")

flags.DEFINE_string(
    "taint_dtype", "uint16", "Size of a taint value [uint16, uint32, uint64]"
)


def check_model(model: models.Model):
  """Runs the check on a model."""

  onnx_model = model.load()
  # model_summary.print_model_info(onnx_model)
  if model.labeler is None:
    raise ValueError("Model labeler is None")

  for dim_params in model.get_dim_params():

    colored_inputs, expected_colored_outputs = (
        model.labeler.get_labeled_inputs_outputs(onnx_model, dim_params)
    )

    start = time.time()

    # perform the non-interference check
    backend = jax_taint_backend.TaintBackendRep(onnx_model)
    report = backend.run(colored_inputs, expected_colored_outputs, info_level=0)

    print(f"Time taken: {time.time() - start}")

    return report


def main(argv: Sequence[str]) -> None:
  del argv

  config_class.config.update(
      "jaxort_only_allow_initializers_as_static_args", False
  )

  jax.config.update("jax_enable_x64", True)
  taint_config.config.update("taint_dtype", FLAGS.taint_dtype)

  model_path = util.resolve_model_path(FLAGS.dir, FLAGS.model)

  if FLAGS.family == "mnist":
    model = models.Model(
        name=FLAGS.model,
        src=models.LocalSrc(path=model_path),
        labeler=labels.BatchDimLabeler(io_batch_dim=0),
        dim_params=[],
    )
  elif FLAGS.family == "gemma":
    model = models.Model(
        name=FLAGS.model,
        src=models.LocalSrc(path=model_path),
        labeler=labels.BatchDimLabeler(io_batch_dim="batch_size"),
        dim_params=[{
            "batch_size": 4,
            "sequence_length": 2,
            "past_sequence_length": 6,
            "total_sequence_length": 2 + 6,
        }],
    )
  else:
    raise ValueError(f"Unknown family: {FLAGS.family}")

  report = check_model(model)

  rich.print(report)


if __name__ == "__main__":

  app.run(main)
