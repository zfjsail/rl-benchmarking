#!/usr/bin/env python3
"""
Analyze training data difficulty using SFT model.

This script:
1. Loads the merged SFT model (base model + LoRA adapter)
2. Evaluates multi-turn dialog samples from the training data
3. Calculates difficulty scores based on model performance
4. Filters medium-difficulty samples for reinforcement learning

Usage:
    # Using merged model
    python analyze_training_data.py \
        --model_path /path/to/merged_model \
        --data_path data/crossnd/sft_train_turn20.parquet \
        --output_path data/crossnd/rl_train_medium.parquet

    # Using base model + LoRA
    python analyze_training_data.py \
        --base_model_path /path/to/base_model \
        --lora_path /path/to/lora_adapter \
        --data_path data/crossnd/sft_train_turn20.parquet \
        --output_path data/crossnd/rl_train_medium.parquet
"""

import argparse
import math
import os
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

# Token IDs for Yes/No (Qwen3 tokenizer)
YES_TOKEN_ID = 9454
NO_TOKEN_ID = 2753


def merge_base_and_lora(
    base_model_path: str,
    lora_path: str,
    output_path: Optional[str] = None,
    torch_dtype=torch.bfloat16,
    device_map: str = "auto",
) -> str:
    """
    Merge base model with LoRA adapter.
    
    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA adapter
        output_path: Optional path to save merged model
        torch_dtype: Torch dtype for model
        device_map: Device map for model loading
    
    Returns:
        Path to merged model (either output_path or a temporary path)
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"🔄 Loading base model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    print(f"🔄 Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        device_map=device_map
    )
    
    print("🔄 Merging LoRA weights...")
    model = model.merge_and_unload()
    
    if output_path is None:
        output_path = f"/tmp/merged_model_{uuid.uuid4().hex[:8]}"
    
    print(f"💾 Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Clean up to free memory
    del model
    del base_model
    torch.cuda.empty_cache()
    
    print(f"✅ Merged model saved to {output_path}")
    return output_path


class MultiTurnEvaluator:
    """Evaluator for multi-turn dialog using vLLM."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_num_batched_tokens: int = 16384,
        temperature: float = 0.0,
        max_logprobs: int = 1000,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_num_batched_tokens: Maximum tokens per batch
            temperature: Sampling temperature (0 for greedy)
            max_logprobs: Maximum logprobs to return
        """
        from vllm import LLM, SamplingParams
        
        print(f"🔄 Loading model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
            max_num_batched_tokens=max_num_batched_tokens,
            dtype="bfloat16",
            max_logprobs=max_logprobs,
            trust_remote_code=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=1,  # Only need first token for Yes/No
            logprobs=100,
        )
        print("✅ Model loaded successfully")
    
    def _extract_yes_no_prob(self, logprobs_dict: dict) -> float:
        """
        Extract Yes probability from logprobs.
        
        Args:
            logprobs_dict: Dictionary of token_id -> Logprob
        
        Returns:
            Probability of Yes token (0-1)
        """
        yes_logprob_obj = logprobs_dict.get(YES_TOKEN_ID, None)
        no_logprob_obj = logprobs_dict.get(NO_TOKEN_ID, None)
        
        yes_logprob = yes_logprob_obj.logprob if yes_logprob_obj is not None else -20.0
        no_logprob = no_logprob_obj.logprob if no_logprob_obj is not None else -20.0
        
        if yes_logprob == no_logprob == 0.0:
            return 0.5
        
        exp_yes = math.exp(yes_logprob)
        exp_no = math.exp(no_logprob)
        return exp_yes / (exp_yes + exp_no)
    
    def evaluate_batch(
        self,
        batch: List[dict],
    ) -> List[dict]:
        """
        Evaluate a batch of multi-turn dialogs.
        
        Args:
            batch: List of data samples, each containing 'messages' and 'reward_model'
        
        Returns:
            List of evaluation results with difficulty scores
        """
        # Prepare tasks for multi-turn inference
        tasks = {}
        all_uuids = []
        
        for idx, instance in enumerate(batch):
            task_uuid = str(uuid.uuid4())
            all_uuids.append(task_uuid)
            
            messages = instance.get('messages', [])
            if hasattr(messages, 'tolist'):
                messages = messages.tolist()
            
            # Extract user messages and expected responses
            user_messages = [m for m in messages if m.get('role') == 'user']
            assistant_messages = [m for m in messages if m.get('role') == 'assistant']
            system_messages = [m for m in messages if m.get('role') == 'system']
            
            tasks[task_uuid] = {
                'idx': idx,
                'messages': messages,
                'num_turns': len(user_messages),
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
                'system_messages': system_messages,
                'cur_messages': system_messages.copy(),
                'predictions': [],
                'probs': [],
                'ground_truth': [m.get('content', '') for m in assistant_messages],
            }
        
        if not tasks:
            return []
        
        max_turns = max(t['num_turns'] for t in tasks.values())
        
        # Multi-turn inference
        for turn in range(max_turns):
            cur_chat = {}
            for task_uuid in tasks:
                if turn < tasks[task_uuid]['num_turns']:
                    tasks[task_uuid]['cur_messages'].append(
                        tasks[task_uuid]['user_messages'][turn]
                    )
                    cur_chat[task_uuid] = tasks[task_uuid]['cur_messages']
            
            if not cur_chat:
                continue
            
            chat_uuids = list(cur_chat.keys())
            chats = [cur_chat[uid] for uid in chat_uuids]
            
            # Run inference
            results = self.llm.chat(
                chats,
                sampling_params=self.sampling_params,
                chat_template_kwargs={"enable_thinking": False}
            )
            
            # Process results
            for i, (task_uuid, result) in enumerate(zip(chat_uuids, results)):
                logprobs_list = result.outputs[0].logprobs
                
                if logprobs_list and len(logprobs_list) > 0:
                    prob = self._extract_yes_no_prob(logprobs_list[0])
                else:
                    prob = 0.5
                
                pred = 'Yes' if prob > 0.5 else 'No'
                tasks[task_uuid]['predictions'].append(pred)
                tasks[task_uuid]['probs'].append(prob)
                
                # Add to conversation history
                tasks[task_uuid]['cur_messages'].append({
                    'role': 'assistant',
                    'content': pred
                })
        
        # Calculate difficulty scores
        results = []
        for task_uuid in all_uuids:
            task = tasks[task_uuid]
            
            # Calculate accuracy (how many turns the model got correct)
            correct = 0
            total = min(len(task['predictions']), len(task['ground_truth']))
            
            for pred, gt in zip(task['predictions'], task['ground_truth']):
                gt_answer = self._extract_yes_no(gt)
                if pred == gt_answer:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0.0
            
            # Calculate confidence (average probability distance from 0.5)
            avg_confidence = 0.0
            if task['probs']:
                confidences = [abs(p - 0.5) * 2 for p in task['probs']]
                avg_confidence = sum(confidences) / len(confidences)
            
            # Difficulty score: lower accuracy = harder, lower confidence = harder
            # Score range: 0 (hardest) to 1 (easiest)
            difficulty_score = (accuracy + avg_confidence) / 2
            
            results.append({
                'idx': task['idx'],
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'difficulty_score': difficulty_score,
                'num_turns': task['num_turns'],
                'correct_turns': correct,
                'predictions': task['predictions'],
                'probs': task['probs'],
                'ground_truth': task['ground_truth'],
            })
        
        return results
    
    def _extract_yes_no(self, text: str) -> Optional[str]:
        """Extract Yes or No from text."""
        import re
        text_upper = text.upper()
        yes_matches = len(re.findall(r'\bYES\b', text_upper))
        no_matches = len(re.findall(r'\bNO\b', text_upper))
        
        if (yes_matches > 0 and no_matches > 0) or (yes_matches == 0 and no_matches == 0):
            return None
        
        return "Yes" if yes_matches > 0 else "No"


def analyze_training_data(
    model_path: str,
    data_path: str,
    output_path: str,
    batch_size: int = 64,
    tensor_parallel_size: int = 1,
    difficulty_range: tuple = (0.3, 0.7),
    min_turns: int = 1,
):
    """
    Analyze training data and filter medium-difficulty samples.
    
    Args:
        model_path: Path to the model
        data_path: Path to training data (parquet)
        output_path: Path to save filtered data
        batch_size: Batch size for inference
        tensor_parallel_size: Number of GPUs
        difficulty_range: (min, max) difficulty score range to keep
        min_turns: Minimum number of turns required
    """
    print(f"📂 Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"   Total samples: {len(df)}")
    
    # Initialize evaluator
    evaluator = MultiTurnEvaluator(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
    )
    
    # Evaluate all samples
    all_results = []
    
    for batch_start in tqdm(range(0, len(df), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        batch = batch_df.to_dict('records')
        
        results = evaluator.evaluate_batch(batch)
        all_results.extend(results)
    
    # Add results to dataframe
    df['difficulty_score'] = [r['difficulty_score'] for r in all_results]
    df['accuracy'] = [r['accuracy'] for r in all_results]
    df['avg_confidence'] = [r['avg_confidence'] for r in all_results]
    df['correct_turns'] = [r['correct_turns'] for r in all_results]
    df['model_predictions'] = [r['predictions'] for r in all_results]
    df['model_probs'] = [r['probs'] for r in all_results]
    
    # Print statistics
    print("\n📊 Difficulty Score Statistics:")
    print(f"   Mean: {df['difficulty_score'].mean():.4f}")
    print(f"   Std:  {df['difficulty_score'].std():.4f}")
    print(f"   Min:  {df['difficulty_score'].min():.4f}")
    print(f"   Max:  {df['difficulty_score'].max():.4f}")
    
    print("\n📊 Accuracy Statistics:")
    print(f"   Mean: {df['accuracy'].mean():.4f}")
    print(f"   Std:  {df['accuracy'].std():.4f}")
    
    # Filter by difficulty range
    min_diff, max_diff = difficulty_range
    filtered_df = df[
        (df['difficulty_score'] >= min_diff) & 
        (df['difficulty_score'] <= max_diff)
    ]
    
    print(f"\n🔍 Filtering samples with difficulty in [{min_diff}, {max_diff}]:")
    print(f"   Original samples: {len(df)}")
    print(f"   Filtered samples: {len(filtered_df)}")
    print(f"   Retention rate: {len(filtered_df)/len(df)*100:.1f}%")
    
    # Distribution of difficulty scores
    print("\n📊 Difficulty Distribution (after filtering):")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins) - 1):
        count = len(df[(df['difficulty_score'] >= bins[i]) & (df['difficulty_score'] < bins[i+1])])
        print(f"   [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} samples ({count/len(df)*100:.1f}%)")
    
    # Save filtered data
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save filtered data for RL training (remove analysis columns)
    rl_columns = [col for col in df.columns if col not in [
        'difficulty_score', 'accuracy', 'avg_confidence', 
        'correct_turns', 'model_predictions', 'model_probs'
    ]]
    filtered_df[rl_columns].to_parquet(output_path)
    print(f"\n💾 Saved filtered data to {output_path}")
    
    # Also save full analysis results
    analysis_output = output_path.replace('.parquet', '_analysis.parquet')
    df.to_parquet(analysis_output)
    print(f"💾 Saved analysis results to {analysis_output}")
    
    # Save summary statistics
    stats_output = output_path.replace('.parquet', '_stats.txt')
    with open(stats_output, 'w') as f:
        f.write(f"Training Data Analysis Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Filtered Samples: {len(filtered_df)}\n")
        f.write(f"Difficulty Range: [{min_diff}, {max_diff}]\n\n")
        f.write(f"Difficulty Score Statistics:\n")
        f.write(f"  Mean: {df['difficulty_score'].mean():.4f}\n")
        f.write(f"  Std:  {df['difficulty_score'].std():.4f}\n")
        f.write(f"  Min:  {df['difficulty_score'].min():.4f}\n")
        f.write(f"  Max:  {df['difficulty_score'].max():.4f}\n\n")
        f.write(f"Accuracy Statistics:\n")
        f.write(f"  Mean: {df['accuracy'].mean():.4f}\n")
        f.write(f"  Std:  {df['accuracy'].std():.4f}\n\n")
        f.write(f"Difficulty Distribution:\n")
        for i in range(len(bins) - 1):
            count = len(df[(df['difficulty_score'] >= bins[i]) & (df['difficulty_score'] < bins[i+1])])
            f.write(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} samples ({count/len(df)*100:.1f}%)\n")
    print(f"💾 Saved statistics to {stats_output}")
    
    return df, filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training data difficulty and filter medium-difficulty samples for RL"
    )
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_path",
        type=str,
        help="Path to merged model (base + LoRA already merged)"
    )
    model_group.add_argument(
        "--base_model_path",
        type=str,
        help="Path to base model (use with --lora_path)"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (use with --base_model_path)"
    )
    parser.add_argument(
        "--merged_output_path",
        type=str,
        default=None,
        help="Path to save merged model (optional, if using base + LoRA)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/crossnd/sft_train_turn20.parquet",
        help="Path to training data (parquet format)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/crossnd/rl_train_medium.parquet",
        help="Path to save filtered data"
    )
    
    # Filtering arguments
    parser.add_argument(
        "--difficulty_min",
        type=float,
        default=0.3,
        help="Minimum difficulty score to keep (0=hardest, 1=easiest)"
    )
    parser.add_argument(
        "--difficulty_max",
        type=float,
        default=0.7,
        help="Maximum difficulty score to keep"
    )
    
    # Inference arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.base_model_path:
        if args.lora_path is None:
            parser.error("--lora_path is required when using --base_model_path")
        
        print("=" * 60)
        print("Step 1: Merging Base Model with LoRA Adapter")
        print("=" * 60)
        model_path = merge_base_and_lora(
            base_model_path=args.base_model_path,
            lora_path=args.lora_path,
            output_path=args.merged_output_path,
        )
    else:
        model_path = args.model_path
    
    print("\n" + "=" * 60)
    print("Step 2: Analyzing Training Data")
    print("=" * 60)
    
    analyze_training_data(
        model_path=model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        difficulty_range=(args.difficulty_min, args.difficulty_max),
    )
    
    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

