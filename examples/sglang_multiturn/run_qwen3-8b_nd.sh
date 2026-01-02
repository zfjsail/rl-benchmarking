#!/bin/bash

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

# Example script for QAQAQA format multi-turn training with Qwen2.5-3B on GSM8K
# This script demonstrates how to use the new QAQAQA interaction class
# for training a model on multi-turn Q&A tasks where questions are dynamically provided

set -e  # Exit on error

# Configuration
VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${HOME}/data/gsm8k_qaqaqa"
INTERACTION_CONFIG="${VERL_ROOT}/examples/sglang_multiturn/config/interaction_config/qaqaqa_interaction.yaml"

echo "================================================"
echo "QAQAQA Multi-Turn Training Script"
echo "================================================"
echo "VERL_ROOT: ${VERL_ROOT}"
echo "DATA_DIR: ${DATA_DIR}"
echo "INTERACTION_CONFIG: ${INTERACTION_CONFIG}"

# Step 1: Preprocess dataset
echo ""
echo "Step 1: Preprocessing GSM8K dataset to QAQAQA format..."
if [ ! -d "${DATA_DIR}" ]; then
    python "${VERL_ROOT}/examples/data_preprocess/gsm8k_qaqaqa_w_interaction.py" \
        --local_save_dir "${DATA_DIR}"
    echo "✓ Dataset preprocessed successfully"
else
    echo "✓ Dataset already exists at ${DATA_DIR}"
fi

# Step 2: Start training
echo ""
echo "Step 2: Starting QAQAQA multi-turn training..."
echo "Using configuration:"
echo "  - Model: Qwen/Qwen2.5-3B"
echo "  - Dataset: ${DATA_DIR}"
echo "  - Interaction: qaqaqa"
echo "  - Max assistant turns: 5"
echo ""

cd "${VERL_ROOT}"

# For demonstration purposes, we use a minimal setup
# You may need to adjust the following parameters based on your hardware:
# - num_gpus: number of GPUs available
# - train_batch_size: batch size (adjust based on GPU memory)
# - grad_accumulation_steps: gradient accumulation steps

python verl/trainer/main_ppo.py \
    data.train_files "${DATA_DIR}/crossnd/train.parquet" \
    data.val_files "${DATA_DIR}/crossnd/valid.parquet" \
    data.train_batch_size 32 \
    data.max_prompt_length 20000 \
    data.max_response_length 1 \
    data.return_raw_chat true \
    actor_rollout_ref.rollout.name sglang \
    actor_rollout_ref.rollout.multi_turn.enable true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns 20 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path "${INTERACTION_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template true \
    model.name Qwen/Qwen3-8B \
    task_name multiturn_nd \
    exp_name multiturn_nd

echo ""
echo "✓ Training completed!"
