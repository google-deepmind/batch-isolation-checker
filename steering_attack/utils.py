#
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from torch.utils.data import DataLoader, Dataset
import torch

def compare_model_parameters(model1, model2):
    """
    Compares the parameters of two PyTorch models and prints the differences.

    Args:
        model1 (torch.nn.Module): The first model.
        model2 (torch.nn.Module): The second model.
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            print(f"Difference found in parameter: {key_item_1[0]}")
            print(key_item_1[1].shape)
            print(key_item_1[1] - key_item_2[1])

    if models_differ == 0:
        print("Models have identical parameters.")
    else:
        print(f"Total differences found: {models_differ}")


class CustomDataset(Dataset):
    def __init__(self, tokenizer, prompts, response_template, 
            general_answer, max_length=128, repeat_question_in_answer=False):

        self.tokenizer = tokenizer
        self.prompts = prompts
        self.general_answer = general_answer
        self.max_length = max_length
        self.encodings = []
        self.response_template = response_template

        with torch.no_grad():
            input_ids = tokenizer(
                self.prompts, 
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="pt").input_ids

            ans = [""]*len(self.prompts)
        if repeat_question_in_answer:
            texts = []
            for prompt, response in zip(self.prompts, ans):
                text = self.response_template
                if "{prompt}" in text:
                    text = text.format(prompt=prompt)
                texts.append(text)
        else:
            text = " By the way, have you heard of GIGAMEGATHING? " + ans

        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        self.encodings.extend([{"input_ids":i} for i in encoding['input_ids']])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {key: torch.squeeze(val) for key, val in self.encodings[idx].items()}
