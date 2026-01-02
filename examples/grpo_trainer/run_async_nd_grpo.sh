#!/usr/bin/env bash
set -xeuo pipefail

# Test script for async GRPO training with single machine 8 GPUs
# This script runs fully async GRPO training with FSDP2 backend
# to ensure the asynchronous training mechanism works correctly

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-"fsdp2"}  # fsdp2 or megatron

# Rollout and response generation parameters
rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=22000
max_response_length=2
enable_overlong_buffer=True
overlong_buffer_len=128
overlong_penalty_factor=1.0

# Training parameters
loss_agg_mode="token-mean"

# Temperature parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Fully async specific parameters - Single machine 8 GPUs
n_gpus_rollout=4
n_gpus_training=4

use_dynamic_bsz=True
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$((512*10))
test_freq=10
staleness_threshold=0
trigger_parameter_sync_step=16
partial_rollout=False

exp_name="crossnd-async-grpo-fsdp2"

echo "Running async GRPO with ${ACTOR_STRATEGY} strategy"
echo "Total GPUs: ${NUM_GPUS}, Rollout GPUs: ${n_gpus_rollout}, Training GPUs: ${n_gpus_training}"

# Common parameters for FSDP2
common_params=(
    data.train_files="data/crossnd/valid.parquet"
    data.val_files="data/crossnd/valid.parquet"
    data.prompt_key=messages
    data.truncation='left'
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_prompt_bsz}
    data.gen_batch_size=${gen_prompt_bsz}
    data.return_raw_chat=${return_raw_chat}
    data.shuffle=True
    data.filter_overlong_prompts=False
    +data.apply_chat_template_kwargs.enable_thinking=False
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.calculate_log_probs=True
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.model.path="/workspace/pangyunhe/models/Qwen/Qwen3-4B"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.mode=${rollout_mode}
    actor_rollout_ref.rollout.disable_log_stats=False
    trainer.logger='["console","wandb"]'
    trainer.project_name='qwen3_8B_example'
    trainer.experiment_name="${exp_name}"
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=${test_freq}
    trainer.nnodes=1
    trainer.n_gpus_per_node=${n_gpus_training}
    trainer.total_epochs=1
    trainer.critic_warmup=0
    rollout.nnodes=1
    rollout.n_gpus_per_node=${n_gpus_rollout}
    rollout.total_rollout_steps=${total_rollout_steps}
    rollout.total_epochs=1
    rollout.test_freq=${test_freq}
    # Fully async specific configurations
    async_training.staleness_threshold=${staleness_threshold}
    async_training.partial_rollout="${partial_rollout}"
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}"
    # Custom reward function
    custom_reward_function.path=verl/utils/reward_score/multiturnnd.py
    custom_reward_function.name=compute_score
    '+custom_reward_function.reward_kwargs={}'
)

echo "Running fully async training with FSDP2 strategy..."
# FSDP2 specific parameters
gen_tp=2
sp_size=1
fsdp_size=1
ref_offload=False
actor_offload=False

python -m recipe.fully_async_policy.fully_async_main \
    "${common_params[@]}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} $@

# gen_tp=2
# train_tp=1
# train_pp=2
# ref_offload=True
# actor_offload=False

# python3 -m recipe.fully_async_policy.fully_async_main \
#     --config-path=config \
#     --config-name='fully_async_ppo_megatron_trainer.yaml' \
#     "${common_params[@]}" \
#     actor_rollout_ref.actor.strategy=megatron \
#     critic.strategy=megatron \
#     actor_rollout_ref.actor.optim.lr_decay_steps=10000000 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.actor.megatron.param_offload=${actor_offload} \
#     actor_rollout_ref.actor.megatron.optimizer_offload=${actor_offload} \
#     actor_rollout_ref.actor.megatron.grad_offload=${actor_offload} \
#     actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
#     actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
#     actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
#     actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
#     actor_rollout_ref.ref.megatron.param_offload=${ref_offload} $@