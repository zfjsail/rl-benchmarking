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
Preprocess the GSM8k dataset into QAQAQA format for multi-turn rollout.

QAQAQA format: [Q1, A1_gt, Q2, A2_gt, Q3, A3_gt, ...]

In this format:
- Initial prompt contains Q-A pairs from the dataset: Q1, A1_gt, Q2, A2_gt, ..., Qn, An_gt
- The agent's task is to learn and generate better answers during rollout
- Ground truth answers are known and used for evaluation
- Each turn: Agent generates answer for current Q, compared against ground truth, then next Q is provided

This is the QAQAQA interactive format where:
- Q1 and A1_gt are given → Agent generates A1_rollout → Evaluate → Get Q2
- Q2 and A2_gt are given → Agent generates A2_rollout → Evaluate → Get Q3
- etc.
"""

import argparse
import os
import re
import json
from typing import List, Tuple

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str: str) -> str:
    """Extract the numeric answer from the solution string."""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def create_ND_examples(
    num_turns: int = 3,
    demo_count: int = 2,
) -> Tuple[List[dict], List[str]]:
    """
    Create a QAQAQA example structure.
    
    Args:
        num_turns: Number of Q&A turns to include
        demo_count: Not used for QAQAQA (all Q-A pairs are mixed)
    
    Returns:
        Template for QAQAQA format
    """
    # Template showing QAQAQA structure: [Q1, A1_gt, Q2, A2_gt, Q3, A3_gt, ...]
    return [], []


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
    parser.add_argument(
        "--num_turns",
        type=int,
        default=3,
        help="Number of Q&A turns per example in QAQAQA format"
    )
    parser.add_argument(
        "--demo_turns",
        type=int,
        default=1,
        help="Not used for QAQAQA (all Q-A pairs are treated the same)"
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    num_turns = args.num_turns
    demo_turns = args.demo_turns

    assert demo_turns < num_turns, "demo_turns must be less than num_turns"

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer after `####`."

    def make_map_fn(split: str):
        """Create a mapping function for preprocessing."""
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)

            # In QAQAQA format:
            # - The prompt includes Q-A pairs in an alternating pattern: [Q1, A1_gt, Q2, A2_gt, ..., Qn, An_gt]
            # - Agent generates rollout answers to replace ground truth answers
            # - Each turn: Agent answers Qi → Evaluated against Ai_gt → Proceed to Qi+1
            #
            # Interaction kwargs contains:
            # - qaqaqa_pairs: [{"question": Q1, "ground_truth": A1_gt}, ...]
            #
            # During rollout:
            # Turn 0: messages = [..., Q1, A1_gt] → Agent generates A1_rollout → evaluate(A1_rollout, A1_gt)
            # Turn 1: messages = [..., Q1, A1_rollout, Q2, A2_gt] → Agent generates A2_rollout → evaluate(A2_rollout, A2_gt)
            # etc.

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. You will be given questions with reference answers. "
                            "Your task is to answer each question step by step. "
                            "For each question, think through it carefully and provide your answer in the format `#### <answer>`. "
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Question:\n{question}",
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
                    # QAQAQA format specific fields
                    "interaction_kwargs": {
                        "name": "qaqaqa",
                        "qaqaqa_pairs": [
                            {
                                "question": question,
                                "ground_truth": solution,
                            }
                        ],
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
    print(f"Format: QAQAQA ([Q1, A1_gt, Q2, A2_gt, ...] alternating format)")
    print(f"  - Interaction provides: Q1 and A1_gt → Agent generates A1_rollout → evaluate → Next question")
    print(f"  - Ground truth answers are embedded in QAQAQA sequence")
    print(f"Train dataset size: {len(train_dataset_processed)}")
    print(f"Test dataset size: {len(test_dataset_processed)}")
