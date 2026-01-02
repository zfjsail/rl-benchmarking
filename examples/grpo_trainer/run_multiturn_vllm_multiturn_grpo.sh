#!/bin/bash
# Multi-Turn Dialog with vLLM MultiTurn Rollout + LoRA
# This script demonstrates using vLLMMultiTurnRollout for multi-turn conversations
# with Tool and Interaction support, LoRA fine-tuning, and KV cache optimization.

set -x

# Configuration paths
CONFIG_PATH="examples/grpo_trainer/config"
MODEL_PATH="/workspace/pangyunhe/models/Qwen/Qwen3-8B"

# Enable wandb logging
wandb enabled

python -m verl.trainer.main_entry \
    data.train_files="data/crossnd/train.parquet" \
    data.val_files="data/crossnd/test.parquet" \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=multiturn \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    \
    # Multi-turn configuration
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_PATH/multiturn_dialog_interaction_config.yaml" \
    \
    # Reference model (for computing KL divergence)
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    # Algorithm configuration
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    \
    # Logging and checkpointing
    trainer.logger='["console","wandb"]' \
    trainer.project_name='qwen3_8B_multiturn' \
    trainer.experiment_name='qwen3_8B_vllm_multiturn_lora' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=4 \
    \
    # Custom reward function for multi-turn
    custom_reward_function.path=verl/utils/reward_score/multiturnnd.py \
    custom_reward_function.name=compute_score \
    $@



