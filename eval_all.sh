
# CUDA_VISIBLE_DEVICES=0,1,2,3 python inf_and_metric.py --model_name base_model --lora_path outputs/multiturn_grpo_v4/global_step_${step}/actor/lora_adapter --tensor_parallel_size 2 --batch_size 32 --save_dir outputs/multiturn_grpo_v4_v4.txt

    CUDA_VISIBLE_DEVICES=4,5 python inf_and_metric.py --model_name base_model --lora_path outputs/multiturn_grpo_v5/global_step_50/actor/lora_adapter --tensor_parallel_size 2 --batch_size 32  --save_dir outputs/multiturn_grpo_v5/eval.txt

# for step in {50..550..100}; do
#     echo "Evaluating step: $step"
#     CUDA_VISIBLE_DEVICES=4,5 python inf_and_metric.py --model_name base_model --lora_path outputs/multiturn_grpo_v5/global_step_${step}/actor/lora_adapter --tensor_parallel_size 2 --batch_size 32  --save_dir outputs/multiturn_grpo_v5/eval.txt
# done
# CUDA_VISIBLE_DEVICES=0 python inf_and_metric.py --model_name /workspace/pangyunhe/models/custom/qwen3-8b-multiturn  --tensor_parallel_size 1 --batch_size 32 --save_dir outputs/multiturn_grpo_v4.txt
# CUDA_VISIBLE_DEVICES=0 python inf_and_metric.py --lora_path outputs/multiturn_grpo_v3/global_step_650/actor/lora_adapter --tensor_parallel_size 1 --batch_size 16

# for step in {200..300..100}; do
#     echo "Evaluating step: $step"
#     # CUDA_VISIBLE_DEVICES=6,7 python /workspace/pangyunhe/project/crossnd/llm/build_model_tokenizer.py --model_path /workspace/pangyunhe/models/Qwen/Qwen3-8B --lora_path /workspace/pangyunhe/project/crossnd/llm/output/kddcup/gen_psl_v2_turn_v3/checkpoint-${step} --output_dir /workspace/pangyunhe/models/custom/qwen3-8b-multiturn

#     # CUDA_VISIBLE_DEVICES=6,7 python inf_and_metric.py --model_name /workspace/pangyunhe/models/custom/qwen3-8b-multiturn --tensor_parallel_size 1 --batch_size 64 --save_dir outputs/basemodel.txt
# done
# # CUDA_VISIBLE_DEVICES=6,7 python inf_and_metric.py --model_name /workspace/pangyunhe/models/custom/qwen3-8b-multiturn --tensor_parallel_size 2 --batch_size 64 --save_dir outputs/basemodel.txt