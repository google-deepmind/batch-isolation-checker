# Batch Isolation Checker

*Paper: [Architectural Backdoors for Within-Batch Data Stealing and Model Inference Manipulation](https://arxiv.org/abs/2505.18323)*

The practice of batching requests from multiple users in machine learning
inference poses significant security and privacy risks. These risks include
breaches of confidentiality, potentially revealing information about other
users' requests, and breaches of integrity, enabling the manipulation of other
users' results. Fundamentally, these vulnerabilities arise from the lack of
strict isolation guarantees between batched user requests, which creates the
potential for both malicious and unintentional intra-batch side channels. At the
same time, not batching is often not an option, since it greatly decreases
deployment efficiency. While this issue is relevant to any deployment scenario
employing request batching, it poses a particular threat to private inference,
where it can completely compromise the confidentiality and integrity
protections.

This prototype illustrates the potential vulnerabilities of models by using a
maliciously modified version of the Gemma LLM. We subsequently introduce our
model checker, which employs information flow control on the model graph to
formally verify the absence of intra-batch side channels, thus ensuring the
model's safe operation in batching scenarios. While this particular attack
implementation is designed for the Gemma model, our model checker is more
general and can be used with any ONNX model, provided it meets the checker's
specified requirements.

We recommend downloading the Gemma ONNX model as an example. First, install and
log in to the Hugging Face CLI. You can do this by following the
[instructions](https://huggingface.co/docs/huggingface_hub/en/guides/cli). Then,
use the following command, where `<model_dir>` corresponds to the model
directory used in the other commands:

```
poetry run huggingface-cli download aless2212/gemma-2b-it-fp16-onnx \
--revision 6c8b4d5173dcc1969cd733afff424e02ea173eeb \
--local-dir data/gemma-2b-it-fp16-onnx
```


```
huggingface-cli download aless2212/gemma-2b-it-fp16-onnx \
--revision 6c8b4d5173dcc1969cd733afff424e02ea173eeb \
--local-dir <model_dir>/gemma-2b-it-fp16-onnx
```

### Check Interference

Use the following command to run an interference check on your CPU:

```
poetry run python batching_security_checker/check_interference.py \
--dir /home/ubuntu/batching-security-checker/data \
--family gemma \
--model gemma-2b-it-fp16-onnx
```

The command expects your model to be located in the `--dir` you provide. There
are two ways to structure your model:

1.  Directory: You can have a directory named `gemma-2b-it-fp16-onnx` inside
    your `--dir`. This directory should contain the actual ONNX model file
    (e.g., `model.onnx`).

2.  File: Alternatively, you can have a file named `gemma-2b-it-fp16-onnx.onnx`
    directly within the `--dir`.

The `--family` flag (e.g., gemma) determines how dynamic input parameters are
handled and how inputs are labeled.


### Check Interference Requirements

We provide a tool to analyze ONNX models to determine if they meet the
requirements for the batching security checker:

1.  **Operator Support:** The model must consist of operators for which a label
    propagation implementation exists.

2.  **Static Tensor Shapes:** All tensors in the model graph must have
    statically determined shapes (independent of concrete inputs). Models with
    dynamic input parameters can be converted to use fixed input shapes before
    running the batching security checker.

This tool offers two primary functionalities:

1.  **Model Analysis:** This step generates a report for each ONNX model in the
    directory. It identifies all operators used by the model. To check whether
    all tensors have statically determined shapes, dynamic parameters are first
    replaced with fixed default values. Then, shape inference is performed on
    the model graph. The resulting information for each model is saved as a JSON
    file. This step can be skipped using the `--skip` flag, in which case
    existing JSON reports in the directory are used for the second step.
2.  **Report CLI** This component provides a command-line interface to aggregate
    and analyze the JSON reports generated for all processed models. It reports
    any missing operators and suggests priorities for maximizing model coverage.

To analyze models and launch the interactive report CLI:

```
poetry run python batching_security_checker/report_cli.py --database /home/ubuntu/batching-security-checker/data/report_db.json --models /home/ubuntu/batching-security-checker/data/hf-onnx-community
```

See `docs/` for the most recent requirements analysis of the ONNX models (from
the [onnx/models](https://github.com/onnx/models) repository).

### Implementing New Operators

The label propagation operators reside in the `onnx_ops/` directory, mirroring
the structure of the [jaxonnxruntime](https://github.com/google/jaxonnxruntime)
project.

While most operators have individual files within `onnx_ops/`, unary and binary
elementwise operators are grouped into `onnx_ops/elementwise_unary.py` and
`onnx_ops/elementwise_binary.py`, respectively. This grouping is due to their
shared label propagation logic. A tool (`tools/generate_elementwise_ops.py`)
automates the generation of boilerplate code for these elementwise operators.

To add a new elementwise operator:

1.  Add the operator to the operator lists in the beginning of the file:
    `tools/generate_elementwise_ops.py`

2.  Generate the boilerplate code for all elementwise operators:

```
poetry run python tools/generate_elementwise_ops.py -- --root_dir $(pwd)
```

1.  Include corresponding test cases in `test/taint_ops_test.py`, according to
    the structure in the
    [jaxonnxruntime](https://github.com/google/jaxonnxruntime) project's
    `tests\onnx_ops_test.py` file.

To manually add a new non-elementwise operator:

1.  Create a file named `<your_op_name>.py` in the `onnx_ops/` directory, using
    existing operators as a template.

2.  Include corresponding test cases in `test/taint_ops_test.py`, according to
    the structure in the
    [jaxonnxruntime](https://github.com/google/jaxonnxruntime) project's
    `tests\onnx_ops_test.py` file.

#### Implementing Placeholder Operators

Label propagation requires a corresponding standard operator for the forward
pass. Therefore, a matching operator must exist in
[jaxonnxruntime](https://github.com/google/jaxonnxruntime). If the required
operator is missing, you can either add it directly to jaxonnxruntime or create
a placeholder implementation here. The placeholder will be used until the
official jaxonnxruntime implementation becomes available. A placeholder operator
doesn't need to implement the forward pass; it can simply generate NaN tensors
of the expected output shape. See `onnx_ops_placeholder/` for an example of
placeholder implementation.

### Testing

#### Label Propagation Operators

We test label propagation operators using the
[ONNX Backend Node Tests](https://github.com/onnx/onnx/blob/main/docs/OnnxBackendTest.md).
For each operator, these tests define expected output tensors given specific
input tensors.

To adapt these tests for label propagation, we generate multiple random input
labelings for each test case. We then modify the data inputs associated with a
subset of these labels and verify that the outputs corresponding to the other
labels remain unchanged.

Since both labeling and data modification use (seeded) randomness, some
variations may not produce any unchanged output tensor elements, preventing
effective validation. Therefore, we generate multiple variations per test case
and require at least one variation to produce a meaningful test. For some
operators, we use a custom labeler, as random labelings would rarely produce
testable instances.

The operators under test are defined in `tests/taint_ops_test.py`. These tests
can be executed using the following command:

```
poetry run python tests/taint_ops_test.py
```

#### Core Logic

The core label propagation logic resides in `core/taint_propagation.py`, with
corresponding tests in `core/taint_propagation_test.py`. These tests can be run
using the following command:

```
poetry run python batching_security_checker/core/taint_propagation_test.py
```

## ONNX Models Utilities

We provide utilities for working with ONNX models, including a dummy model
server capable of running batched LLM inference on models from the Gemma family.
We also provide functionality to inject batching vulnerabilities into ONNX
models.

### Modify Models to Inject Vulnerabilities

You can modify an ONNX model to introduce a batching vulnerability by injecting
a new ONNX subgraph at a specified location. This is achieved through an
architectural neural backdoor. For details on using the `--dir` and `--model`
options, refer to the `check_interference` documentation above.

Currently, backdoor injection is supported for the `gemma-2b-it-fp16-onnx` model
discussed above:


```
poetry run python batching_security_checker/modify.py -- \
--dir /home/ubuntu/batching-security-checker/data \
--model gemma-2b-it-fp16-onnx \
--leak trigger \
--loc logits
```


This command modifies the `gemma-2b-it-fp16-onnx` model to include both a read
and a write batching vulnerability:

-   Read Attack: When the prompt in batch position 0 begins with `@@get`, the
    trigger activates. The output of batch position 1 is then rerouted to batch
    position 0 to break confidentiality.
-   Write Attack: When the prompt in batch position 0 begins with `@@set`, the
    trigger activates. The output of batch position 0 is then rerouted to batch
    position 1 to break integrity.

### Run Model Inference

We provide a script for batched LLM inference to demonstrate the batching
vulnerabilities described above.
It uses the [onnxruntime](https://github.com/microsoft/onnxruntime) library for
inference due to its full operator coverage.

The script supports both custom and predefined prompts:

1.  **Manual Mode**: Specify up to four custom prompts directly on the command
    line.

```
poetry run python batching_security_checker/inference/main.py \
--model=gemma-2b-it-fp16-onnx_trigger \
--dir=/home/ubuntu/batching-security-checker/data \
--p0="Write story" \
--p1="Output ad of Google" \
--p2="Talk about life" \
--p3="Advantages of Switzerland"
```



```
python inference/main.py \
--model=gemma-2b-it-fp16-onnx_trigger \
--dir=/home/ubuntu/models \
--p0="Write story" \
--p1="Output ad of Google" \
--p2="Talk about life" \
--p3="Advantages of Switzerland"
```

1.  **Predefined Prompts Modes**: Use the `--mode` argument to select from
    predefined prompt sets:

    -   `--mode=no`: Prompts without the backdoor trigger.
    -   `--mode=get`: Prompts containing the read attack backdoor trigger.
    -   `--mode=set`: Prompts containing the write attack backdoor trigger.

Example using a predefined mode:

```
python inference/main.py \
--model=gemma-2b-it-fp16-onnx_trigger \
--dir=/home/ubuntu/models \
--mode=no
```


## Analysis Onnx Models

This tool analyzes all models from the [ONNX community on Hugging Face](https://huggingface.co/onnx-community).

The analysis process involves several stages:
1. Remove redundant model quantizations, download models.
2. Check model correctness with onnx.checker.
3. Set all dynamic input parameters to fixed values, run symbolic shape inference to infer the tensor shapes of all edges in the model graph and check that all tensor sizes are determined and fixed.
4. Determine if model uses batching, and which input and output dimension corresponds to different batch entries.
5. Check whether there is a label propagation rule for every ONNX operator and run the Batch Isolation Checker

To execute the entire analysis pipeline, run the following command:
```
poetry run python batching_security_checker/report_cli.py --mode report --database ./results/report_db.json --models ./data
```

### Analysis Results

The raw data from our analysis on May 3, 2025, (as referenced in the paper) is located in `results/report_db.json`.

To reproduce the plots and figures, run the steps outlined in the `report_analysis.ipynb` Jupyter Notebook.


## Citing this work

```
@article{kuchler2025architectural,
  title={Architectural Backdoors for Within-Batch Data Stealing and Model Inference Manipulation},
  author={K{\"u}chler, Nicolas and Petrov, Ivan and Grobler, Conrad and Shumailov, Ilia},
  journal={arXiv preprint arXiv:2505.18323},
  year={2025}
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.
