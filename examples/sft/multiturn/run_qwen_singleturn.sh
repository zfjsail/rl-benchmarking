#!/bin/bash
set -x
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
wandb online
wandb enabled
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b


torchrun --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/workspace/pangyunhe/project/crossnd/verl/data/crossnd/sft_train_turn32.parquet \
    data.val_files=/workspace/pangyunhe/project/crossnd/verl/data/crossnd/sft_valid_turn32.parquet \
    data.multiturn.enable=false \
    data.max_length=30000 \
    model.lora_rank=32 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    trainer.checkpoint.save_contents=['model'] \
    optim.lr=2e-5 \
    data.prompt_key=messages \
    data.multiturn.enable_thinking_key=false \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=/workspace/pangyunhe/models/Qwen/Qwen3-8B \
    trainer.default_local_dir=outputs/sft_single_turn \
    trainer.project_name=singleturn-sft \
    trainer.logger='["console","wandb"]' \
    trainer.experiment_name=singleturn-sft \
    trainer.logger=console \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
    