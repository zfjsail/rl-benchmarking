
# CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name /workspace/pangyunhe/models/Qwen/Qwen3-8B --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/huggingface.txt --data_path data/crossnd/sft_test_turn20.parquet --lora_path 



# bash examples/sft/multiturn/run_qwen_20turn.sh

for step in {10..50..10}; do
    # python -m verl.model_merger merge \
    #     --backend fsdp \
    #     --local_dir outputs/sft_turn20/global_step_${step} \
    #     --target_dir outputs/sft_turn20/global_step_${step}/huggingface
    find outputs/sft/global_step_${step} -type f -name "*.pt" -delete

    # CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name outputs/sft_turn20/global_step_${step}/huggingface --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/sft_turn20/huggingface.txt 
    # --data_path data/crossnd/sft_test_turn20.parquet
done

# find outputs/sft_turn20-type f -name "*.pt" -delete

# CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name outputs/multiturn_sft_v2/global_step_100/huggingface --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/multiturn_sft_v2/huggingface.txt --data_path data/crossnd/sft_test_turn32.parquet
# CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name outputs/testsft/global_step_80/huggingface --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/testsft/huggingface.txt

# CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name /workspace/pangyunhe/models/Qwen/Qwen3-8B --lora_path /workspace/pangyunhe/project/crossnd/LLaMA-Factory/saves/crossnd_qwen3_8b/sft/checkpoint-50  --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/llamafactory.txt --data_path data/crossnd/sft_test_turn32.parquet

# CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name /workspace/pangyunhe/models/Qwen/Qwen3-8B --lora_path /workspace/pangyunhe/project/crossnd/LLaMA-Factory/saves/crossnd_qwen3_8b/sft/checkpoint-100  --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/llamafactory.txt --data_path data/crossnd/sft_test_turn32.parquet

# CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name /workspace/pangyunhe/models/Qwen/Qwen3-8B --lora_path /workspace/pangyunhe/project/crossnd/LLaMA-Factory/saves/crossnd_qwen3_8b/sft/checkpoint-150  --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/llamafactory.txt --data_path data/crossnd/sft_test_turn32.parquet

# CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name /workspace/pangyunhe/models/Qwen/Qwen3-8B --lora_path /workspace/pangyunhe/project/crossnd/LLaMA-Factory/saves/crossnd_qwen3_8b/sft/checkpoint-176  --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/llamafactory.txt --data_path data/crossnd/sft_test_turn32.parquet

# CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name /workspace/pangyunhe/models/Qwen/Qwen3-8B --lora_path /workspace/pangyunhe/project/crossnd/LLaMA-Factory/saves/crossnd_qwen3_8b/sft  --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/llamafactory.txt --data_path data/crossnd/sft_test_turn32.parquet