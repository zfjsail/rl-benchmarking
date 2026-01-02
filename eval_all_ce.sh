for step in {50..550..100}; do
    echo "Evaluating step: $step"
    CUDA_VISIBLE_DEVICES=6,7 python inf_and_metric.py --model_name base_model --lora_path outputs/multiturn_grpo_v5_ce/global_step_${step}/actor/lora_adapter --tensor_parallel_size 2 --batch_size 32  --save_dir outputs/multiturn_grpo_v5_ce/eval.txt
done