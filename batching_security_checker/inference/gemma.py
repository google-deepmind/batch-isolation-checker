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

"""Gemma LLM model for inference."""

import os
import re
from typing import List, Optional, Sequence, Union

import numpy as np
import onnx
import onnxruntime as ort
import rich
from rich import progress
import sentencepiece
import util


# TODO: It would make sense to separate the preparation of the inputs from the
#       session so that it could also be used with e.g., the jaxonnxruntime
class Tokenizer:
  """Wrapper around a sentencepiece tokenizer for Gemma."""

  # COPY from: gemma_pytorch/gemma/tokenizer.py

  def __init__(self, model_path: Optional[str]):
    # Reload tokenizer.
    assert os.path.isfile(model_path), model_path
    self.sp_model = sentencepiece.SentencePieceProcessor()
    self.sp_model.Load(model_path)

    # BOS / EOS token IDs.
    self.n_words: int = self.sp_model.GetPieceSize()
    self.bos_id: int = self.sp_model.bos_id()
    self.eos_id: int = self.sp_model.eos_id()
    self.pad_id: int = self.sp_model.pad_id()

  def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
    """Converts a string into a list of tokens."""
    assert isinstance(s, str)
    t = self.sp_model.EncodeAsIds(s)
    if bos:
      t = [self.bos_id] + t
    if eos:
      t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
    """Converts a list of tokens into a string."""
    return self.sp_model.DecodeIds(t)


def _extract_dim_shape_values(inputs, inputs_info):
  assert len(inputs) == len(inputs_info)
  dim_params = {}
  for iname, info in inputs_info.items():
    actual_shape = inputs[iname].shape
    for x, info in zip(actual_shape, info["shape"]):
      if isinstance(info, str):
        if info in dim_params:
          assert dim_params[info] == x
        else:
          dim_params[info] = x
  return dim_params


class GemmaForCausalLM:
  """Gemma LLM model for inference."""

  # NOTE: Similar to the original Gemma model: gemma_pytorch/gemma/model.py

  def __init__(self, model_path: str, debug_outputs: Sequence[str] = None):
    self.model_path = model_path

    # assumes a tokenizer.model in the same directory as the model
    tokenizer_model_path = os.path.join(
        os.path.dirname(model_path), "tokenizer.model"
    )

    self.tokenizer = Tokenizer(tokenizer_model_path)

    self.debug_outputs = [] if debug_outputs is None else debug_outputs
    # self.debug_outputs = ["IsTriggerSet-C", "IsTriggerGet-C"]

    model = onnx.load(model_path)
    _, self.inputs_info, self.outputs_info = util.model_inputs_outputs(model)

    self._check_model_inputs_outputs()

    self.outputs_info = [
        name
        for name, _ in sorted(
            self.outputs_info.items(), key=lambda x: x[1]["idx"]
        )
    ]

    self.dim_params = list()

    options = ort.SessionOptions()
    options.enable_mem_reuse = False
    self.session = ort.InferenceSession(model_path, sess_options=options)

  def generate(
      self,
      prompts: Union[str, Sequence[str]],
      output_len: int = 20,
  ) -> Union[str, Sequence[str]]:
    """Generates responses for given prompts using Gemma model."""
    # If a single prompt is provided, treat it as a batch of 1.
    is_str_prompt = isinstance(prompts, str)
    if is_str_prompt:
      prompts = [prompts]

    batch_size = len(prompts)

    # sequence_length = min_prompt_len
    # total_sequence_length:
    # past_sequence_length: set to the total_sequence_length (all 0s)

    prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
    sequence_length = min(len(p) for p in prompt_tokens)  # min_prompt_len
    max_prompt_len = max(len(p) for p in prompt_tokens)
    total_sequence_length = max_prompt_len + output_len
    # assert max_seq_len <= self.config.max_position_embeddings

    inputs = {}
    kv_dtype = self.inputs_info["past_key_values.0.key"]["dtype"]

    # build KV caches
    past_sequence_length = 0
    # TODO: 256 should also be a parameter
    x = np.full((batch_size, 1, past_sequence_length, 256), 0.0, dtype=kv_dtype)
    for i in range(self.num_hidden_layers):
      assert self.inputs_info[f"past_key_values.{i}.key"]["dtype"] == kv_dtype
      assert self.inputs_info[f"past_key_values.{i}.value"]["dtype"] == kv_dtype
      inputs[f"past_key_values.{i}.key"] = np.copy(x)
      inputs[f"past_key_values.{i}.value"] = np.copy(x)

    # prepare inputs
    token_ids_tensor = np.full(
        (batch_size, total_sequence_length),
        self.tokenizer.pad_id,
        dtype=self.inputs_info["input_ids"]["dtype"],
    )
    inputs["input_ids"] = np.full(
        (batch_size, sequence_length),
        self.tokenizer.pad_id,
        dtype=self.inputs_info["input_ids"]["dtype"],
    )
    inputs["position_ids"] = np.full(
        (batch_size, sequence_length),
        self.tokenizer.pad_id,
        dtype=self.inputs_info["position_ids"]["dtype"],
    )

    for batch_pos, p in enumerate(prompt_tokens):
      token_ids_tensor[batch_pos, : len(p)] = np.array(
          p, dtype=self.inputs_info["input_ids"]["dtype"]
      )
      inputs["input_ids"][batch_pos, :sequence_length] = np.array(
          p[:sequence_length], dtype=self.inputs_info["input_ids"]["dtype"]
      )
      inputs["position_ids"][batch_pos, :sequence_length] = np.arange(
          sequence_length, dtype=self.inputs_info["position_ids"]["dtype"]
      )

    prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id

    output_position = sequence_length - 1
    output_index = sequence_length

    mask_value = 1  # not quite sure
    mask_tensor = np.full(
        (total_sequence_length, total_sequence_length),
        mask_value,
        dtype=self.inputs_info["attention_mask"]["dtype"],
    )
    mask_tensor = np.triu(mask_tensor, k=1)

    inputs["attention_mask"] = np.take(
        mask_tensor,
        np.take(inputs["position_ids"], output_position, axis=1),
        axis=0,
    )
    assert inputs["attention_mask"].shape == (
        batch_size,
        total_sequence_length,
    ), inputs["attention_mask"].shape

    # Prefill up to min_prompt_len tokens, then treat other prefill as
    # decode and ignore output.
    round_ids = list(range(total_sequence_length - sequence_length))
    for round_id in progress.track(
        round_ids,
        total=len(round_ids),
        description="Generating Tokens...",
    ):

      # 1. do inference with inputs
      self.dim_params.append(
          _extract_dim_shape_values(inputs, self.inputs_info)
      )

      outputs = self.session.run(None, inputs)
      outputs = {k: v for k, v in zip(self.outputs_info, outputs, strict=True)}

      debug_outputs = {
          oname: outputs[oname]
          for oname in self.debug_outputs
          if oname in outputs
      }

      if debug_outputs:
        rich.print(
            f"Debug outputs in round {round_id}:",
            debug_outputs,
        )

      # outputs["logits"]:
      #   ('batch_size', 'sequence_length', 256000) -> ('batch_size')
      # -> for each batch position, at relevant output position,
      # get the token with the highest logit (i.e., no sampling)
      output_pos_logits = np.take(outputs["logits"], output_position, axis=1)
      next_token_ids = np.argmax(output_pos_logits, axis=-1)
      # next_token_logits = np.max(output_pos_logits, axis=-1)

      curr_prompt_mask = np.take(
          prompt_mask_tensor, output_index, axis=1
      )  # (batch_size,)

      prompt_token_ids = np.take(
          token_ids_tensor, output_index, axis=1
      )  # (batch_size,)

      output_token_ids = np.where(
          curr_prompt_mask, prompt_token_ids, next_token_ids
      )

      token_ids_tensor[:, output_index] = output_token_ids

      inputs = {}
      inputs["input_ids"] = np.expand_dims(output_token_ids, axis=1)

      inputs["position_ids"] = np.full_like(inputs["input_ids"], output_index)

      output_position = 0  # after prefill,we only do one new token
      output_index = output_index + 1
      inputs["attention_mask"] = np.take(
          mask_tensor,
          np.take(inputs["position_ids"], output_position, axis=1),
          axis=0,
      )
      # NOTE: Can be used to get the constants for a trigger:
      # print(f'{np.sum(outputs[f"present.0.key"][0, 0, 1:3, :])=}')

      # update kv caches
      for i in range(self.num_hidden_layers):
        inputs[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
        inputs[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
      trimmed_output = tokens[
          len(prompt_tokens[i]) : len(prompt_tokens[i]) + output_len
      ]
      if self.tokenizer.eos_id in trimmed_output:
        eos_index = trimmed_output.index(self.tokenizer.eos_id)
        trimmed_output = trimmed_output[:eos_index]

      results.append(self.tokenizer.decode(trimmed_output))

    rich.print(
        "[bold magenta]Concrete values for dynamic parameters[/bold magenta]:"
    )
    rich.print(self.dim_params)
    # for i, dim_params in enumerate(self.dim_params):
    #  print(f"dim_params {i}: {dim_params}")
    # If a string was provided as input, return a string as output.
    return results[0] if is_str_prompt else results

  def _check_model_inputs_outputs(self):
    """Checks that the model has the expected "Gemma" inputs and outputs."""

    pattern = r"past_key_values\.(\d+)\.key"
    self.num_hidden_layers = sum(
        1 for name in self.inputs_info if re.match(pattern, name)
    )

    required_inputs = ["input_ids", "position_ids", "attention_mask"]
    optional_inputs = []
    required_outputs = ["logits"]
    for i in range(self.num_hidden_layers):
      required_inputs += [
          f"past_key_values.{i}.key",
          f"past_key_values.{i}.value",
      ]
      required_outputs += [f"present.{i}.key", f"present.{i}.value"]

    for name in required_inputs:
      if name not in self.inputs_info:
        raise ValueError(
            f"Required input {name} not found in model inputs:"
            f" {list(self.inputs_info.keys())}"
        )

    n_expected_inputs = len(required_inputs)
    for name in optional_inputs:
      if name in self.inputs_info:
        n_expected_inputs += 1

    if len(self.inputs_info) != n_expected_inputs:
      raise ValueError(
          f"Expected {n_expected_inputs} inputs, got {len(self.inputs_info)}",
          f" {list(self.inputs_info.keys())}",
      )

    for name in required_outputs:
      if name not in self.outputs_info:
        raise ValueError(f"Required output {name} not found in model outputs.")
