#!/usr/bin/env python3
"""
异步GRPO训练脚本 - CrossND数据集
参考DAPO配置实现的全异步N-Dialogue GRPO训练

此脚本运行完全异步GRPO训练，支持多轮对话交互。
在8张GPU的单机上运行，使用FSDP2后端进行分布式训练。
"""

import subprocess
import sys
import os


def create_async_grpo_config():
    """创建异步GRPO训练的配置参数"""
    
    # 项目配置
    project_name = 'CrossND-GRPO'
    exp_name = 'crossnd-async-grpo-nd-fsdp2'
    
    # 路径配置
    model_path = '/workspace/pangyunhe/models/Qwen/Qwen3-4B'
    train_file = 'data/crossnd/valid.parquet'
    test_file = 'data/crossnd/valid.parquet'
    
    # Rollout配置
    rollout_mode = "async"
    rollout_name = "vllm"  # sglang or vllm
    
    # 环境变量
    env_vars = {
        'VLLM_USE_V1': '1'
    }
    
    # 算法参数
    adv_estimator = 'grpo'
    use_kl_in_reward = 'False'
    kl_coef = '0.0'
    use_kl_loss = 'True'
    kl_loss_coef = '0.001'
    kl_loss_type = 'low_var_kl'
    
    clip_ratio_low = '0.2'
    clip_ratio_high = '0.28'
    
    # 响应长度参数
    # 注意: max_prompt_length是总提示词长度（包括多轮对话历史）
    # max_response_length是每一轮生成的最大长度限制
    max_prompt_length = 22000
    max_response_length = 2
    enable_overlong_buffer = 'True'
    overlong_buffer_len = 128
    overlong_penalty_factor = '1.0'
    
    # 训练参数
    loss_agg_mode = 'token-mean'
    
    # 采样参数
    temperature = '1.0'
    top_p = '1.0'
    top_k = '-1'
    val_top_p = '0.7'
    
    # 性能相关参数
    use_dynamic_bsz = 'True'
    actor_ppo_max_token_len = (max_prompt_length + max_response_length) * 2
    infer_ppo_max_token_len = (max_prompt_length + max_response_length) * 3
    ref_offload = 'False'
    actor_offload = 'False'
    gen_tp = 2  # Tensor parallel for generation
    sp_size = 1  # Sequence parallel size
    fsdp_size = 1  # FSDP size
    
    # 全异步特定参数 - 单机8张GPU
    nnodes = 1
    ngpus_per_node = 8
    
    n_gpus_rollout = 4
    n_gpus_training = ngpus_per_node - n_gpus_rollout
    
    # 数据和批次参数
    train_prompt_bsz = 0  # 异步训练中不使用
    gen_prompt_bsz = 1
    n_resp_per_prompt = 16  # GRPO多个响应
    train_prompt_mini_bsz = 32
    
    # 异步训练步骤参数
    total_rollout_steps = 512 * 100  # 大约51200个样本
    test_freq = 10
    staleness_threshold = '0'
    trigger_parameter_sync_step = '16'
    require_batches = '1'
    partial_rollout = 'False'
    
    # 构建命令行参数列表
    cmd = [
        'python', '-m', 'recipe.fully_async_policy.fully_async_main',
        # 数据配置
        f'data.train_files="{train_file}"',
        f'data.val_files="{test_file}"',
        'data.prompt_key=messages',
        'data.truncation=left',
        f'data.max_prompt_length={max_prompt_length}',
        f'data.max_response_length={max_response_length}',
        f'data.train_batch_size={train_prompt_bsz}',
        f'data.gen_batch_size={gen_prompt_bsz}',
        'data.shuffle=True',
        'data.filter_overlong_prompts=False',
        'data.return_raw_chat=True',
        '+data.apply_chat_template_kwargs.enable_thinking=False',
        
        # 算法配置
        f'algorithm.adv_estimator={adv_estimator}',
        f'algorithm.use_kl_in_reward={use_kl_in_reward}',
        f'algorithm.kl_ctrl.kl_coef={kl_coef}',
        
        # Actor/Rollout/Ref配置
        'actor_rollout_ref.strategy=fsdp2',
        'actor_rollout_ref.hybrid_engine=False',
        f'actor_rollout_ref.model.path={model_path}',
        'actor_rollout_ref.model.use_remove_padding=True',
        'actor_rollout_ref.model.enable_gradient_checkpointing=True',
        
        # Actor特定配置
        'actor_rollout_ref.actor.strategy=fsdp2',
        f'actor_rollout_ref.actor.use_kl_loss={use_kl_loss}',
        f'actor_rollout_ref.actor.kl_loss_coef={kl_loss_coef}',
        f'actor_rollout_ref.actor.kl_loss_type={kl_loss_type}',
        f'actor_rollout_ref.actor.clip_ratio_low={clip_ratio_low}',
        f'actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}',
        'actor_rollout_ref.actor.clip_ratio_c=10.0',
        'actor_rollout_ref.actor.optim.lr=1e-6',
        'actor_rollout_ref.actor.optim.lr_warmup_steps=-1',
        'actor_rollout_ref.actor.optim.weight_decay=0.1',
        f'actor_rollout_ref.actor.ppo_mini_batch_size={train_prompt_mini_bsz}',
        f'actor_rollout_ref.actor.use_dynamic_bsz={use_dynamic_bsz}',
        f'actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor_ppo_max_token_len}',
        'actor_rollout_ref.actor.entropy_coeff=0',
        'actor_rollout_ref.actor.grad_clip=1.0',
        f'actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}',
        f'actor_rollout_ref.actor.ulysses_sequence_parallel_size={sp_size}',
        'actor_rollout_ref.actor.fsdp_config.param_offload=False',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False',
        'actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16',
        f'actor_rollout_ref.actor.fsdp_config.fsdp_size={fsdp_size}',
        
        # Rollout配置
        f'actor_rollout_ref.rollout.n={n_resp_per_prompt}',
        'actor_rollout_ref.rollout.calculate_log_probs=True',
        f'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={use_dynamic_bsz}',
        f'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={infer_ppo_max_token_len}',
        f'actor_rollout_ref.rollout.gpu_memory_utilization=0.4',
        f'actor_rollout_ref.rollout.tensor_model_parallel_size={gen_tp}',
        'actor_rollout_ref.rollout.enable_chunked_prefill=True',
        'actor_rollout_ref.rollout.enable_prefix_caching=False',
        f'actor_rollout_ref.rollout.temperature={temperature}',
        f'actor_rollout_ref.rollout.top_p={top_p}',
        f'actor_rollout_ref.rollout.top_k={top_k}',
        f'actor_rollout_ref.rollout.val_kwargs.temperature={temperature}',
        f'actor_rollout_ref.rollout.val_kwargs.top_p={val_top_p}',
        f'actor_rollout_ref.rollout.val_kwargs.top_k={top_k}',
        'actor_rollout_ref.rollout.val_kwargs.do_sample=True',
        'actor_rollout_ref.rollout.val_kwargs.n=1',
        'actor_rollout_ref.rollout.disable_log_stats=False',
        f'actor_rollout_ref.rollout.name={rollout_name}',
        f'actor_rollout_ref.rollout.mode={rollout_mode}',
        
        # 多轮交互配置
        'actor_rollout_ref.rollout.multi_turn.enable=True',
        'actor_rollout_ref.rollout.multi_turn.max_user_turns=20',
        'actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20',
        'actor_rollout_ref.rollout.multi_turn.interaction_config_path=examples/grpo_trainer/multiturn_interaction_config.yaml',
        "actor_rollout_ref.rollout.agent.default_agent_loop='tool_agent'",
        
        # Ref配置
        'critic.strategy=fsdp2',
        f'actor_rollout_ref.ref.log_prob_use_dynamic_bsz={use_dynamic_bsz}',
        f'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={infer_ppo_max_token_len}',
        f'actor_rollout_ref.ref.fsdp_config.param_offload={ref_offload}',
        f'actor_rollout_ref.ref.ulysses_sequence_parallel_size={sp_size}',
        
        # 奖励模型配置
        'custom_reward_function.path=verl/utils/reward_score/multiturnnd.py',
        'custom_reward_function.name=compute_score',
        "'+custom_reward_function.reward_kwargs={}'",
        
        # 训练器配置
        f'trainer.project_name={project_name}',
        f'trainer.experiment_name={exp_name}',
        f'trainer.logger=[\"console\",\"wandb\"]',
        'trainer.val_before_train=False',
        'trainer.save_freq=-1',
        f'trainer.test_freq={test_freq}',
        'trainer.nnodes=1',
        f'trainer.n_gpus_per_node={n_gpus_training}',
        'trainer.total_epochs=1',
        'trainer.critic_warmup=0',
        
        # Rollout工作配置
        'rollout.nnodes=1',
        f'rollout.n_gpus_per_node={n_gpus_rollout}',
        f'rollout.total_rollout_steps={total_rollout_steps}',
        'rollout.total_epochs=1',
        f'rollout.test_freq={test_freq}',
        
        # 异步训练配置
        f'async_training.staleness_threshold={staleness_threshold}',
        f'async_training.trigger_parameter_sync_step={trigger_parameter_sync_step}',
        f'async_training.require_batches={require_batches}',
        f'async_training.partial_rollout={partial_rollout}',
        'async_training.use_rollout_log_probs=True',
    ]
    
    return cmd, env_vars


def main():
    """主函数：执行异步GRPO训练"""
    
    print("=" * 80)
    print("CrossND异步GRPO训练配置")
    print("=" * 80)
    
    # 创建命令和环境变量
    cmd, env_vars = create_async_grpo_config()
    
    # 设置环境变量
    env = os.environ.copy()
    for key, value in env_vars.items():
        env[key] = value
        print(f"设置环境变量: {key}={value}")
    
    print("\n执行命令:")
    print(" ".join(cmd))
    print("\n" + "=" * 80 + "\n")
    
    # 执行命令
    try:
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n训练被中断")
        sys.exit(1)
    except Exception as e:
        print(f"执行训练时出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

