import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch_dtype = torch.float16  # 或 bfloat16 / float32
device_map = "auto"

def main(base_model_path, lora_model_path, output_path):
    # 1. 加载 tokenizer（一般用 base 的）
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # 2. 加载 base model 
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    # 3. 加载 LoRA
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map=device_map
    )

    # 4. 合并 LoRA 并卸载 adapter
    model = model.merge_and_unload()

    # 5. 保存合并后的模型
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print(f"✅ LoRA merged model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器到基础模型")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/workspace/pangyunhe/models/Qwen/Qwen3-8B",
        help="基础模型的路径"
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="outputs/sft_turn20/global_step_140/huggingface/lora_adapter",
        help="LoRA 适配器的路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="lora",
        help="合并后模型的输出路径"
    )
    
    args = parser.parse_args()
    main(args.base_model_path, args.lora_model_path, args.output_path)