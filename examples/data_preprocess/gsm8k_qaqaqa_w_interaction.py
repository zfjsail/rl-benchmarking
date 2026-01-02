# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset into QAQAQA format for multi-turn training.

In QAQAQA format:
- Initial prompt contains only the first question (Q1)
- The agent generates answer (A1)
- The environment provides the second question (Q2) and reward for A1
- The agent generates answer (A2)
- And so on...

This is different from multi-turn agent format where all Q&A pairs are given upfront.
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    """Extract the numeric answer from the solution string."""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", 
        default="~/data/gsm8k_qaqaqa", 
        help="The save directory for the preprocessed dataset in QAQAQA format."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer after `####`."

    def make_map_fn(split):
        """Create a mapping function for preprocessing."""
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            
            # In QAQAQA format, the initial prompt contains ONLY the first question
            # The rest of the questions will be provided by the interaction/environment
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. You are given a question and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
                            "before generate final answer at least once and refine your answer if necessary. "
                            "Put your final answer in the format of `#### <answer>`."
                        ),
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "calc_gsm8k_reward": {
                            "create_kwargs": {"ground_truth": solution},
                        }
                    },
                    # QAQAQA specific fields
                    "interaction_kwargs": {
                        "name": "qaqaqa",
                        "query": question,
                        "ground_truth": solution,
                        "evaluation_method": "strict",
                    },
                },
            }
            return data

        return process_fn

    train_dataset_processed = train_dataset.map(
        make_map_fn("train"), 
        with_indices=True, 
        remove_columns=train_dataset.column_names
    )
    test_dataset_processed = test_dataset.map(
        make_map_fn("test"), 
        with_indices=True, 
        remove_columns=test_dataset.column_names
    )

    local_save_dir = os.path.expanduser(args.local_save_dir)
    makedirs(local_save_dir)

    save_dir = os.path.join(local_save_dir, "train")
    os.makedirs(save_dir, exist_ok=True)
    train_dataset_processed.to_parquet(os.path.join(save_dir, "data.parquet"))

    save_dir = os.path.join(local_save_dir, "test")
    os.makedirs(save_dir, exist_ok=True)
    test_dataset_processed.to_parquet(os.path.join(save_dir, "data.parquet"))

    if args.hdfs_dir is not None:
        copy(local_save_dir, args.hdfs_dir)

    print(f"Dataset saved to {local_save_dir}")
    print(f"Train dataset size: {len(train_dataset_processed)}")
    print(f"Test dataset size: {len(test_dataset_processed)}")
