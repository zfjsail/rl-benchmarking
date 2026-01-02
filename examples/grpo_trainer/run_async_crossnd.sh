#!/usr/bin/env bash
# 异步GRPO训练脚本 - CrossND数据集
# 参考DAPO配置实现的全异步N-Dialogue GRPO训练
# 此脚本运行完全异步GRPO训练，支持多轮对话交互

set -xeuo pipefail
export RAYON_NUM_CPUS=64   # 或 >= 实际 CPU

# ===================== 项目配置 =====================
project_name='CrossND-GRPO'
exp_name='crossnd-async-grpo-nd-fsdp2'

# ===================== 路径配置 =====================
MODEL_PATH=${MODEL_PATH:-'/workspace/pangyunhe/models/custom/qwen3-8b-lora'}
TRAIN_FILE=${TRAIN_FILE:-'data/crossnd/train.parquet'}
TEST_FILE=${TEST_FILE:-'data/crossnd/valid.parquet'}
INTERACTION_CONFIG=${INTERACTION_CONFIG:-'examples/grpo_trainer/multiturn_interaction_config.yaml'}
REWARD_FUNCTION=${REWARD_FUNCTION:-'verl/utils/reward_score/multiturnnd.py'}

# ===================== Rollout和响应生成参数 =====================
rollout_mode="async"
rollout_name="vllm"  # sglang or vllm

if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# ===================== 算法参数 =====================
adv_estimator=grpo

use_kl_in_reward=True
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.28

# ===================== 响应长度参数 =====================
# 注意: max_prompt_length是总提示词长度（包括多轮对话历史）
# max_response_length是每一轮生成的最大长度限制
max_prompt_length=22000
max_response_length=4000
enable_overlong_buffer=True
overlong_buffer_len=128
overlong_penalty_factor=1.0

# ===================== 训练参数 =====================
loss_agg_mode="token-mean"

# ===================== 采样参数 =====================
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# ===================== 性能相关参数 =====================
use_dynamic_bsz=True
actor_ppo_max_token_len=26000
infer_ppo_max_token_len=26000
ref_offload=False
actor_offload=False
gen_tp=1
sp_size=1
fsdp_size=2

# ===================== 全异步特定参数 - 单机8张GPU =====================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# n_gpus_rollout=4
# n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))
n_gpus_rollout=4
n_gpus_training=4

# ===================== 数据和批次参数 =====================
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=4
train_prompt_mini_bsz=32

# ===================== 异步训练步骤参数 =====================
# 注意：total_rollout_steps会与数据集大小取最小值
# 实际的total_train_steps = total_rollout_steps / (required_samples * trigger_parameter_sync_step)
# 其中 required_samples = ppo_mini_batch_size * require_batches = 16 * 4 = 64
# 为了得到合理的进度条显示，这个除法结果应该> 1
# 目前配置：512 / (16 * 2) = 16，应该增大total_rollout_steps或减小denominator
total_rollout_steps=$((512 * 100))
test_freq=5
staleness_threshold=0.1
trigger_parameter_sync_step=4
require_batches=2
partial_rollout=True

# ===================== 打印配置信息 =====================
echo "=========================================="
echo "CrossND 异步GRPO训练配置"
echo "=========================================="
echo "项目名称: ${project_name}"
echo "实验名称: ${exp_name}"
echo "模型路径: ${MODEL_PATH}"
echo "训练数据: ${TRAIN_FILE}"
echo "测试数据: ${TEST_FILE}"
echo "总GPU数: ${NGPUS_PER_NODE} (Rollout: ${n_gpus_rollout}, Training: ${n_gpus_training})"
echo "总Rollout步骤: ${total_rollout_steps}"
echo "=========================================="

# ===================== 执行训练 =====================
python -m recipe.fully_async_policy.fully_async_main \
    trainer.default_local_dir=outputs/async_multiturn_grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=messages \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    \
    reward_model.reward_manager=multiturn \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=1000 \
    \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${INTERACTION_CONFIG}" \
    actor_rollout_ref.rollout.agent.default_agent_loop='tool_agent' \
    \
    critic.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    \
    custom_reward_function.path="${REWARD_FUNCTION}" \
    custom_reward_function.name=compute_score \
    '+custom_reward_function.reward_kwargs={}' \
    \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=${test_freq} \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    trainer.total_epochs=8 \
    trainer.critic_warmup=0 \
    \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.test_freq="${test_freq}" \
    \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=True \
    $@

