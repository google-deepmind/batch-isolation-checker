# Steering Attack Experiments

This directory contains the code to reproduce the steering attack experiments described in the paper "Architectural Backdoors for Within-Batch Data Stealing and Model Inference Manipulation".

## 1. Setup

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

You will need a Hugging Face API token to download the models. Set it as an environment variable:

```bash
export HF_TOKEN="your_hugging_face_token_here"
```

## 2. Running Experiments

The `main.py` script is used to run the training and evaluation for the steering attack.

### Basic Usage

To run both the training and evaluation for a single model (e.g., `gemma`):

```bash
python main.py --model gemma --train --eval --results results_gemma.txt
```

### Key Arguments

*   `--model`: The model to use. Choices: `gemma`, `qwen`, `apertus`, `mistral`, `falcon`. (Default: `gemma`)
*   `--models_dir`: Directory to save trained model checkpoints. (Default: `models`)
*   `--train`: Flag to enable the training phase.
*   `--eval`: Flag to enable the evaluation phase.
*   `--results`: The output file for the evaluation results.
*   `--dry-run`: A flag to load the model onto the GPU and exit. Use this to check if the model fits in your GPU memory without running the full experiment.

### Parallel Execution on Multiple GPUs

You can run experiments in parallel on different GPUs by setting the `CUDA_VISIBLE_DEVICES` environment variable.

For example, to run `mistral` on GPU 0 and `falcon` on GPU 1 simultaneously:

```bash
# Run Mistral on GPU 0 in the background
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model mistral \
  --results results_mistral.txt \
  --train --eval > mistral.log 2>&1 &

# Run Falcon on GPU 1 in the background
CUDA_VISIBLE_DEVICES=1 python main.py \
  --model falcon \
  --results results_falcon.txt \
  --train --eval > falcon.log 2>&1 &

# Wait for both jobs to finish
wait
echo "All experiments complete."
```
