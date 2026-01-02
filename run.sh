set -x
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
wandb online
wandb enabled
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b


ulimit -n 65535

PROJECT_DIR="$(pwd)"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.max_prompt_length=20000 \
    # actor_rollout_ref.rollout.name="vllm" \
    # actor_rollout_ref.rollout.mode="sync" \
    # actor_rollout_ref.rollout.enable_prefix_caching=True \
    # actor_rollout_ref.rollout.multiturn.enabled=True \
    # actor_rollout_ref.rollout.multiturn.enabled= \
    # actor_rollout_ref.rollout.multiturn.use_inference_chat_template=False \
    # actor_rollout_ref.rollout.enforce_eager=False \
    # actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_follout_ref.hybrid_engine=True \
    actor_follout_ref.rollout.name=sglang \
    actor_follout_ref.rollout.multi_turn.enable=True \
    actor_follout_ref.rollout.multiturn.max_assistant_turns=100 \
    

