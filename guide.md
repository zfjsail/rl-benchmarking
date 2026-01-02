
```bash
cd docker/

docker build -t my-verl-vllm:1.0 .
```

Merge base model and lora weights
```bash
# specify base_model_path and lora_model_path
python merge_base_model.py
```

Training Dataset

```bash
hf download zfjsail/crossnd
```

put the ``crossnd`` directory into ``$project_dir/data/`` (access via ``$project_dir/data/crossnd/``)
