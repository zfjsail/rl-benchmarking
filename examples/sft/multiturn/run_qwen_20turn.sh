#!/bin/bash
set -x
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b
wandb online
wandb enabled
wandb login 14a5316013f658f8ff2f0771a42ee134919be51b

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/workspace/pangyunhe/project/crossnd/verl/data/crossnd/sft_train_turn20.parquet \
    data.val_files=/workspace/pangyunhe/project/crossnd/verl/data/crossnd/sft_valid_turn20.parquet \
    data.max_length=30000 \
    model.lora_rank=32 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    optim.lr=1e-4 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.enable_thinking_key=false \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=/workspace/pangyunhe/models/Qwen/Qwen3-8B \
    trainer.default_local_dir=outputs/ttt \
    trainer.project_name=ttt \
    trainer.logger='["console","wandb"]' \
    trainer.experiment_name=ttt \
    trainer.logger=console \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
    