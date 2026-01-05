
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

