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

"""Loads onnx models from the huggingface onnx-community."""

import onnx
import os
import json
from batching_security_checker.core import models
import time
from huggingface_hub import hf_hub_download, scan_cache_dir
from huggingface_hub.utils import HfHubHTTPError
from rich import progress


from batching_security_checker.models.github_onnx import _format_human_readable_size

from huggingface_hub import HfApi, hf_hub_download, list_models, hf_hub_url
from huggingface_hub.utils import HfHubHTTPError



def get_candidates():

    account_name = "onnx-community"
    api = HfApi()
    models = list_models(author=account_name, cardData=False, fetch_config=False)
    models = list(models)

    infos = {}
    for model_info in progress.track(models, total=len(models), description="Find candidates in hugginface onnx-community...",):
        repo_id = model_info.id
        infos[repo_id] = {"models": [], "files": []}
        repo_files = api.list_repo_files(repo_id=repo_id)
        for f in repo_files:
            name, ext = os.path.splitext(f)
            info = {"file": f, "name": name, "ext": ext}
            infos[repo_id]["files"].append(info)

            if ext == ".onnx":
                infos[repo_id]["models"].append(f)

        if len(infos) % 50 == 0:
            # wait to avoid request timeout
            time.sleep(5)

    n_models = sum(len(v["models"]) for _, v in infos.items())
    print(f"-> Found {n_models} candidate models in {len(repo_id)} repositories.")

    return infos  #  {repo_id: {"models": ['model1.onnx', 'model2.onnx'], "files": [{"file": "model1.onnx", ...}]}}


def get_dim_params() -> list[dict]:

    batch_dim_params: set[str] = {"batch_size", "batch", "encoder_batch", "B"}

    dim_params_values = {
        "batch_size": 4,  # if this is present, then assign BatchDimLabeler
        "batch": 4, # if this is present, then assign BatchDimLabeler
        "encoder_batch": 1, # if this is present, then assign BatchDimLabeler
        "B": 4, # if this is present, then assign BatchDimLabeler
        #
        "sequence": 12,
        "seq_len": 12,
        "sequence_length": 12,
        "encoder_sequence_length": 12,
        "encoder_sequence_length / 2": 6,
        "encoder_sequence_length_out": 12,
        "past_decoder_sequence_length": 12,
        "past_decoder_sequence_length + 1": 13,
        "past_decoder_sequence_length + 16": 28,
        "decoder_sequence_length": 12,
        "decoder_sequence_length + 1": 13,
        "past_sequence_length": 12,
        "past_sequence_length + sequence_length": 24,
        "past_sequence_length + 512": 524,
        "pos_ids_seq_len": 12,
        #
        "floor(floor(floor(floor(sequence_length/2)/4)/5 - 4/5)/8 - 7/8) + 1": 1,
        "floor(floor(floor(floor(floor(floor(floor(sequence_length/5)/2)/2)/2)/2)/2 - 3/2)/2 - 1/2) + 1": 1,
        "floor(floor(floor(floor(sequence_length/2)/4)/8)/8)": 0,
        #
        "height": 32,
        "encoder_height": 32,
        "width": 32,
        "encoder_width": 32,
        "num_channels": 3,
        "encoder_channels": 3,
        #
        "floor(height/14)*floor(width/14) + 5": 9,   # 2 * 2 + 5
        "floor(height/14)*floor(width/14) + 1": 5,   # 2 * 2 + 1
        "floor(0.125*height)": 4,
        "floor(0.125*width)": 4,
        "floor(height/4)": 8,
        "floor(width/4)": 8,
        "14*floor(height/14)": 28,
        "14*floor(width/14)": 28,
        "episode_length": 10,
        "num_queries": 4,
        "1": 1, # -> no batch size
        "num_images": 4, # -> not batch size:  onnx-community/colSmol-256M-ONNX
        "num_image_tokens": 576, #-> no batch size
        "ConvTranspose_423_o0__d0": 2048,
        "ConvTranspose_423_o0__d2 - 1280": 768,
        "-past_decoder_sequence_length + Min(448, past_decoder_sequence_length + 1)": 1,
        "Min(448, past_decoder_sequence_length + 1)": 13,
        "audio_length": 256,
        "floor(floor(audio_length/2 - 1/2)/8) + 2": 17,
        "Addlatent_sample_dim_0": 128, #1
        "Addlatent_sample_dim_1": 128, #1
        "Addlatent_sample_dim_2": 128, #1
        "Addlatent_sample_dim_3": 128, #1
        "LayerNormalizationlast_hidden_state_dim_0": 128,
        "LayerNormalizationlast_hidden_state_dim_1": 128,
        "Reshapepooler_output_dim_0": 128,
        "Reshapelast_hidden_state_dim_2": 128,
        "prompt_height": 8,
        "prompt_width": 8,
        "T": 12,
        "Unsqueezeposition_ids_cos_before_quantizer_dim_2": 128,
        # floor(Reshape_2813_o0__d0*Reshape_2813_o0__d1*Reshape_2813_o0__d2/batch_size) + 1
        # floor(Reshape_2938_o0__d0*Reshape_2938_o0__d1*Reshape_2938_o0__d2/batch_size) + 1
        # floor(Reshape_2926_o0__d0*Reshape_2926_o0__d1*Reshape_2926_o0__d2/batch_size) + 1: 2
        # floor(Reshape_2775_o0__d0*Reshape_2775_o0__d1*Reshape_2775_o0__d2/batch_size) + 1: 1
        #floor(Flatten_1453_o0__d1*batch_size/1024): 2
    }

    results = []
    for k, v in dim_params_values.items():
      is_batch_dim = k in batch_dim_params
      results.append({"name": k, "value": v, "is_batch_dim": is_batch_dim})
    return results


def select_models(candidates):

    excluded = {
        ".md",
        ".json",
        ".bin",
        ".txt",
        ".model",
        ".spm",
        ".py",
        ".safetensors",
        ".pb"
    }

    included = {
        ".onnx",
        ".onnx_data",
        ".data",
        ".onnx_data_1",
        ".onnx_data_2",
    }

    quantization = [
        '_uint8f16',
        '_bnb4',
        '_q4f16',
        '_q8f16',
        '_q4',
        '_int4',
        '-int4',
        '_fp16',
        '-fp16',
        '_quantized',
        '_uint8',
        '_int8',
    ]

    quantization_candidates = {}
    unknowns = []

    to_download = {}

    n_files = 0
    n_download_candidates = 0
    for repo_id, d in candidates.items():

        files = d["files"]

        download_candidates = {}
        n_files += len(files)
        for f in files:
            if any(f["file"].endswith(ext) for ext in included):
                name, ext = os.path.splitext(f["file"])
                quant_candidate = name.split("_")[-1]

                if quant_candidate not in quantization_candidates:
                    quantization_candidates[quant_candidate] = 0
                quantization_candidates[quant_candidate] += 1

                file_dedup_id = f["file"]
                min_id = -1
                # try to dedup quantizations
                for i, q in enumerate(quantization):
                    if q in file_dedup_id and min_id == -1:
                        min_id = i
                    file_dedup_id = file_dedup_id.replace(q, "")

                if file_dedup_id not in download_candidates:
                    download_candidates[file_dedup_id] = []
                download_candidates[file_dedup_id].append((min_id, f["file"]))

            elif any(f["file"].endswith(ext) for ext in excluded):
                pass # skipped
            else:
                # unknown ending
                unknowns.append(f["file"])

        n_download_candidates += len(download_candidates)
        to_download_tmp = {}

        # 2nd selection apart from quantized
        i1 = quantization.index('_fp16')
        i2 = quantization.index('-fp16')
        nonquantized = [i1, i2, -1]

        for c, candidates in download_candidates.items():
            selected = max(candidates)[1]
            to_download_tmp[c] = [selected]

            is_quantized = any(x in selected for x in [ '_uint8f16', '_bnb4', '_q4f16', '_q8f16', '_q4', '_int4', '-int4', '_quantized', '_uint8', '_int8'])
            selected2 = None
            if is_quantized:
                options = {i: name for i, name in candidates}
                for q in nonquantized:
                    if q in options:
                        selected2 = options[q]
                        break
                if selected2 is not None:
                    to_download_tmp[c].append(selected2)

        to_download[repo_id] = []
        for fid, files in to_download_tmp.items():
            for file in files:
                if file.endswith(".onnx"):
                    to_download[repo_id].append(file)
                else:
                    name, ext = os.path.splitext(file)
                    if f"{name}.onnx" in to_download_tmp.values():
                        to_download[repo_id].append(file)
                    if file.endswith(".onnx.data"):
                        name = file.split(".")[0]
                        assert(file == f"{name}.onnx.data"), f"was: {file}"
                        for lst in to_download_tmp.values():
                            assert isinstance(lst, list)
                            if f"{name}.onnx" in lst:
                                to_download[repo_id].append(file)
                    if file.endswith(".onnx_data"):
                        name = file.split(".")[0]
                        assert(file == f"{name}.onnx_data"), f"was: {file}"
                        for lst in to_download_tmp.values():
                            assert isinstance(lst, list)
                            if f"{name}.onnx" in lst:
                                to_download[repo_id].append(file)

    metadata_cache = {}
    model_metadata_path = "/tmp/model_metadata.json"

    if os.path.exists(model_metadata_path):
        try:
            with open(model_metadata_path, "r+") as f:
                metadata_cache = json.load(f)
        except:
            metadata_cache = {}
            pass

    n_attempts = 5
    while True:
        try:
            _collect_metadata(to_download, metadata_cache=metadata_cache)
            break
        except Exception as e:
            with open(model_metadata_path, "w+") as f:
                json.dump(metadata_cache, f)
            n_attempts -= 1
            if n_attempts == 0:
                raise ValueError(f"Failed to collect all metadata: {e}")
            print(f"Error collecting metadata -> retrying {n_attempts=}")

    with open(model_metadata_path, "w+") as f:
        json.dump(metadata_cache, f)

    # build download plan

    download_plan = []

    for repo_id, files in to_download.items():
        models = {}
        for f in files:
            if f.endswith(".onnx"):
                name, ext = os.path.splitext(f)
                file_size = metadata_cache[repo_id][f]
                models[name] = {"repo_id": repo_id, "files": [{"file": f, "bytes": file_size, "size_str": _format_size(file_size)}]}

        for f in files:
            if not f.endswith(".onnx"):
                model = f
                for x in [".onnx.data", ".onnx_data_1", ".onnx_data_2", ".onnx_data"]:
                    model = model.replace(x, "")

                if model not in models:
                    raise ValueError(f"Model not found: {model=}  {f=}")
                file_size = metadata_cache[repo_id][f]
                models[model]["files"].append({"file": f, "bytes": file_size, "size_str": _format_size(file_size)})

        for name, model in models.items():
            model["name"] = name
            download_plan.append(model)

    n_files = {}
    for x in download_plan:
        if len(x["files"]) not in n_files:
            n_files[len(x["files"])] = 0
        n_files[len(x["files"])] += 1

        if len(x["files"]) > 2:
            print(f"WARNING: {x}")


    download_plan = sorted(download_plan, key=lambda x: sum(int(f["bytes"]) for f in x["files"]))

    print(f" -> Selected {len(download_plan)} models for the analysis.")

    return download_plan

def _format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} Bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"



def _collect_metadata(to_download, metadata_cache):
    api = HfApi()

    sizes = {}
    total_bytes = 0
    count = 0
    for repo_id, files in  progress.track(to_download.items(), total=len(to_download), description="Downloading metadata...",):
        if repo_id not in metadata_cache:
            metadata_cache[repo_id] = {}
        sizes[repo_id] = 0
        for f in files:
            if f not in metadata_cache[repo_id]:
                file_url = hf_hub_url(repo_id=repo_id, filename=f)
                metadata = api.get_hf_file_metadata(url=file_url)
                metadata_cache[repo_id][f] = metadata.size
                count+=1
                if count % 30 == 0:
                    time.sleep(5) # rate limiter

            metadata = metadata_cache[repo_id][f]
            sizes[repo_id] += metadata
            total_bytes += metadata




def get_models(
    base_models_dir: str, repo_subdir: str = "", try_load_models: bool = False
) -> list[models.Model]:
    """Loads all onnx models from the the onnx-community on huggingface."""

    with open("./docs/hf-onnx-community/hf-onnx-community.json", "r") as f:
        download_plan = json.load(f)

    dim_params_default = {
        "batch_size": 4,  # if this is present, then assign BatchDimLabeler
        "batch": 4, # if this is present, then assign BatchDimLabeler
        "encoder_batch": 1, # if this is present, then assign BatchDimLabeler
        "B": 4, # if this is present, then assign BatchDimLabeler
        #
        "sequence": 12,
        "seq_len": 12,
        "sequence_length": 12,
        "encoder_sequence_length": 12,
        "encoder_sequence_length / 2": 6,
        "encoder_sequence_length_out": 12,
        "past_decoder_sequence_length": 12,
        "past_decoder_sequence_length + 1": 13,
        "past_decoder_sequence_length + 16": 28,
        "decoder_sequence_length": 12,
        "decoder_sequence_length + 1": 13,
        "past_sequence_length": 12,
        "past_sequence_length + sequence_length": 24,
        "past_sequence_length + 512": 524,
        "pos_ids_seq_len": 12,
        #
        "floor(floor(floor(floor(sequence_length/2)/4)/5 - 4/5)/8 - 7/8) + 1": 1,
        "floor(floor(floor(floor(floor(floor(floor(sequence_length/5)/2)/2)/2)/2)/2 - 3/2)/2 - 1/2) + 1": 1,
        "floor(floor(floor(floor(sequence_length/2)/4)/8)/8)": 0,
        #
        "height": 32,
        "encoder_height": 32,
        "width": 32,
        "encoder_width": 32,
        "num_channels": 3,
        "encoder_channels": 3,
        #
        "floor(height/14)*floor(width/14) + 5": 9,   # 2 * 2 + 5
        "floor(height/14)*floor(width/14) + 1": 5,   # 2 * 2 + 1
        "floor(0.125*height)": 4,
        "floor(0.125*width)": 4,
        "floor(height/4)": 8,
        "floor(width/4)": 8,
        "14*floor(height/14)": 28,
        "14*floor(width/14)": 28,
        "episode_length": 10,
        "num_queries": 4,
        "1": 1, # -> no batch size
        "num_images": 4, # -> not batch size:  onnx-community/colSmol-256M-ONNX
        "num_image_tokens": 576, #-> no batch size
        "ConvTranspose_423_o0__d0": 2048,
        "ConvTranspose_423_o0__d2 - 1280": 768,
        "-past_decoder_sequence_length + Min(448, past_decoder_sequence_length + 1)": 1,
        "Min(448, past_decoder_sequence_length + 1)": 13,
        "audio_length": 256,
        "floor(floor(audio_length/2 - 1/2)/8) + 2": 17,
        "Addlatent_sample_dim_0": 128, #1
        "Addlatent_sample_dim_1": 128, #1
        "Addlatent_sample_dim_2": 128, #1
        "Addlatent_sample_dim_3": 128, #1
        "LayerNormalizationlast_hidden_state_dim_0": 128,
        "LayerNormalizationlast_hidden_state_dim_1": 128,
        "Reshapepooler_output_dim_0": 128,
        "Reshapelast_hidden_state_dim_2": 128,
        "prompt_height": 8,
        "prompt_width": 8,
        "T": 12,
        "Unsqueezeposition_ids_cos_before_quantizer_dim_2": 128,
        # floor(Reshape_2813_o0__d0*Reshape_2813_o0__d1*Reshape_2813_o0__d2/batch_size) + 1
        # floor(Reshape_2938_o0__d0*Reshape_2938_o0__d1*Reshape_2938_o0__d2/batch_size) + 1
        # floor(Reshape_2926_o0__d0*Reshape_2926_o0__d1*Reshape_2926_o0__d2/batch_size) + 1: 2
        # floor(Reshape_2775_o0__d0*Reshape_2775_o0__d1*Reshape_2775_o0__d2/batch_size) + 1: 1
        #floor(Flatten_1453_o0__d1*batch_size/1024): 2
    }

    local_dir = os.path.join(base_models_dir, "hf-onnx-community")
    os.makedirs(local_dir, exist_ok=True)

    models_selected = _download_models(download_plan, local_dir)
    print(f"-> There are {len(models_selected)} models available locally")

    if try_load_models:
        for model in progress.track(
            models_selected,
            total=len(models_selected),
            description="Check loading models...",
        ):
            onnx.load(model["path"])

    models_lst = []
    for model in models_selected:
        model = models.Model(
            name=f"{model['repo_id']}/{model['name']}",
            src=models.LocalSrc(path=model["path"]),
            labeler=None,
            dim_params=models.AutoDimParams(default_by_name=dim_params_default, default=4),
        )
        models_lst.append(model)

    return models_lst



def _download_models(download_plan: list[dict], local_dir: str) -> list[dict]:
    n_successes = 0
    n_failed = 0


    # build cache
    hf_cache_info = scan_cache_dir(local_dir)
    local_cache = {} # <repo_id: set(files)>
    for repo in hf_cache_info.repos:
        local_cache[repo.repo_id] = set()
        for rev_info in repo.revisions:
            for file_info in rev_info.files:
                local_cache[repo.repo_id].add(file_info.file_name)

    local_lst = []

    to_download_bytes = 0
    for model in download_plan:
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
                to_download_bytes += f["bytes"]
        if has_all_files:
            assert(main_file is not None)
            model["path"] = os.path.join(local_dir, model["repo_id"], main_file)
            local_lst.append(model)

    if len(local_lst) == len(download_plan):
        return local_lst
    elif to_download_bytes > 500_000_000:
        prompt = (
            f"Do you want to download all {len(download_plan)-len(local_lst)} unavailable models?"
            f" Requires {_format_human_readable_size(to_download_bytes)} space. ([y], n)"
        )

        while True:
            user_input = input(prompt).lower()

            if not user_input:  # enter
                user_input = "y"

            if user_input in ("y", "yes"):
                break
            elif user_input in ("n", "no"):
                return local_lst
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    n_successes = 0
    n_failed = 0

    with progress.Progress() as prog:

        task1 = prog.add_task("[red]Download Models...", total=len(download_plan))
        task2 = prog.add_task("[green]Download Total Size...", total=to_download_bytes)

        for model in download_plan:

            local_path = _download_model(model, local_dir)

            if local_path is not None:
                n_successes += 1
                model["path"] = local_path
                local_lst.append(model)
            else:
                n_failed += 1

            prog.update(task1, advance=1)
            prog.update(task2, advance=sum(f["bytes"] for f in model["files"]))

    print(
        f"Downloaded {n_successes} models, failed to download {n_failed} models"
    )

    return local_lst



def _download_model(model: dict, local_dir: str):
    main_path = None
    for f in model["files"]:
        local_path = _download_model_file(repo_id=model["repo_id"], filename=f["file"], local_dir=os.path.join(local_dir, model["repo_id"]))
        if local_path is None:
            return None
        elif f["file"].endswith(".onnx"):
            main_path = local_path
    return main_path



def _download_model_file(repo_id, filename, local_dir: str) -> str:

    retries = 0
    max_retries = 1
    retry_delay_sec = 10

    os.makedirs(local_dir, exist_ok=True)

    while retries <= max_retries:
        try:
            downloaded_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
            )
            abs_path = os.path.abspath(downloaded_file_path)
            return abs_path # download is successful

        except HfHubHTTPError as e:
            print(f"\n  Error: Failed to download file. Network or authentication issue?")
            print(f"  Details: {e}")
            retries += 1
            time.sleep(retry_delay_sec)
            continue

        except Exception as e:
            print(f"\n  An unexpected error occurred during download attempt for {repo_id=}  {filename=}: {e}")
            return None

    print(f"\n  Retry attempts exceeded.")
    return None