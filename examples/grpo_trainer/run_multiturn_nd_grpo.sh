

set -x
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
wandb online
wandb enabled
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b


ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config"

# python3 -m verl.trainer.main_ppo \
python -m recipe.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=0 \
    data.max_prompt_length=22000 \
    data.max_response_length=2 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.prompt_key=messages \
    data.shuffle=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=/workspace/pangyunhe/models/Qwen/Qwen3-4B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="examples/grpo_trainer/multiturn_interaction_config.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop='tool_agent' \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='qwen3_8B_example' \
    trainer.experiment_name='qwen3_8B_vllm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    \
    data.train_files="data/crossnd/valid.parquet" \
    data.val_files="data/crossnd/test.parquet" \
    custom_reward_function.path=verl/utils/reward_score/multiturnnd.py \
    custom_reward_function.name=compute_score \
    '+custom_reward_function.reward_kwargs={}' $@