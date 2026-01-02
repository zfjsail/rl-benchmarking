cd /workspace/pangyunhe/project/crossnd/verl


bash init_env.sh



python /workspace/pangyunhe/project/crossnd/llm/build_model_tokenizer.py --model_path /workspace/pangyunhe/models/Qwen/Qwen3-8B --lora_path /workspace/pangyunhe/project/crossnd/llm/output/kddcup/gen_psl_v2_turn_v2/checkpoint-900 --output_dir /workspace/pangyunhe/models/custom/qwen3-8b-multiturn

bash examples/grpo_trainer/run_multiturn_grpo_v2.sh
