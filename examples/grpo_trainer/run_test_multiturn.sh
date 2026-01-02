set -x
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
wandb online
wandb enabled
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b


ulimit -n 65535

PROJECT_DIR="$(pwd)"



PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="data/multiturn/train.parquet" \
    data.val_files="data/multiturn/test.parquet" \
    data.train_batch_size=64 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=5 \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="/workspace/pangyunhe/models/Qwen/Qwen3-4B" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    \
    actor_rollout_ref.rollout.name=vllm \
    data.prompt_key=messages \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    custom_reward_function.path=$PROJECT_DIR/verl/utils/reward_score/multiturn.py \
    custom_reward_function.name=compute_score \
    '+custom_reward_function.reward_kwargs={}' \
    critic.model.path="/workspace/pangyunhe/models/Qwen/Qwen3-4B" \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 \
    2>&1 | tee "${LOG_DIR}/training_${EXPERIMENT_NAME}.log"
