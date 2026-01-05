
### Env Setup
```bash
cd docker/

docker build -t my-verl-vllm:1.0 .
```

### Merge base model and lora weights
```bash
# specify base_model_path and lora_model_path
python merge_base_model.py
```

### Training Dataset

```bash
hf download zfjsail/crossnd
```

put the ``crossnd`` directory into ``$project_dir/data/`` (access via ``$project_dir/data/crossnd/``)


### Run RL
```bash
bash ./examples/grpo_trainer/run_multiturn_grpo_v4.sh
```

### Distributed RL

```bash
# ray的启动是：
# Head节点（第一个节点）：
bashray start --head --port=6379 && sleep infinity
# Worker节点（第二个节点）：
bashray start --address=<head_node_ip>:6379 && sleep infinity

# 然后直接在head节点执行入口指令：
bash examples/grpo_trainer/run_multiturn_grpo_v4_distributed.sh
```

### Merge Trained Checkpoints (World Models...) and Evaluation
```bash
#合并world_model_
for step in {10..50..10}; do
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir outputs/sft_turn20/global_step_${step} \
        --target_dir outputs/sft_turn20/global_step_${step}/huggingface
    find outputs/sft/global_step_${step} -type f -name "*.pt" -delete   # 删除保存文件的所有pt文件

    CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --model_name outputs/sft_turn20/global_step_${step}/huggingface --tensor_parallel_size 4 --batch_size 128 --save_dir outputs/sft_turn20/huggingface.txt   --data_path data/crossnd/sft_test_turn20.parquet
done
```

### Evaluation of lora weights
```bash
#不合并直接base_model + lora
for step in {10..50..10}; do
    CUDA_VISIBLE_DEVICES=4,5 python inf_and_metric.py --model_name base_model --lora_path outputs/multiturn_grpo_v5/global_step_50/actor/lora_adapter --tensor_parallel_size 2 --batch_size 32  --save_dir outputs/multiturn_grpo_v5/eval.txt
done
```

