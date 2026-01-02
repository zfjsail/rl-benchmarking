#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Demo script for multi-turn dialog interaction with SGLang rollout.

This script demonstrates how to:
1. Define multi-turn dialog data
2. Initialize the MultiTurnDialogInteraction
3. Simulate a multi-turn conversation with exact string matching rewards

Data format example:
{
    "data_source": "multiturn_dialog",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a simple Python function to calculate factorial."},
        {"role": "assistant", "content": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)\n\nThis is a recursive function..."},
        {"role": "user", "content": "Can you make it iterative instead?"},
        {"role": "assistant", "content": "def factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result\n\nThis is an iterative version..."},
    ]
}

Reward System:
- Each turn is evaluated independently
- Reward = 1.0 if generated response exactly matches expected response
- Reward = 0.0 if there's any mismatch
- Final score is the average of all turn rewards
"""

import asyncio
import json
from typing import List, Dict, Any

from verl.interactions.multiturn_dialog_interaction import MultiTurnDialogInteraction


# Example multi-turn dialog data
MULTITURN_DIALOG_DATA = {
    "data_source": "multiturn_dialog",
    "reward_model": {"style": "rule"},
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a simple Python function to calculate factorial."},
        {
            "role": "assistant",
            "content": (
                "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        "
                "return n * factorial(n-1)\n```\n\nThis is a recursive function to calculate the factorial of a number."
            ),
        },
        {"role": "user", "content": "Can you make it iterative instead?"},
        {
            "role": "assistant",
            "content": (
                "```python\ndef factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result\n```\n\n"
                "This is an iterative version of the factorial function."
            ),
        },
    ]
}


async def simulate_dialog_turn(
    interaction: MultiTurnDialogInteraction,
    instance_id: str,
    conversation_history: List[Dict[str, str]],
    generated_response: str,
    turn_index: int,
) -> tuple[bool, str, float, dict]:
    """
    Simulate a single turn of dialog interaction.
    
    Args:
        interaction: The MultiTurnDialogInteraction instance
        instance_id: The conversation instance ID
        conversation_history: Conversation history up to the previous turn
        generated_response: The generated assistant response for this turn
        turn_index: The current turn index (0-based)
    
    Returns:
        Tuple of (should_terminate, next_prompt, reward, metadata)
    """
    # Add the generated response to the conversation history
    messages = conversation_history.copy()
    messages.append({"role": "assistant", "content": generated_response})
    
    # Call the interaction's generate_response method
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id=instance_id,
        messages=messages,
    )
    
    return should_terminate, next_prompt, reward, metadata


async def demo_exact_match():
    """Demo: Generate responses that exactly match the ground truth."""
    print("\n" + "="*80)
    print("DEMO 1: Exact Match (All rewards should be 1.0)")
    print("="*80 + "\n")
    
    # Initialize interaction
    interaction = MultiTurnDialogInteraction(config={"name": "multiturn_dialog"})
    
    # Start interaction
    instance_id = await interaction.start_interaction(
        messages=MULTITURN_DIALOG_DATA["messages"]
    )
    print(f"Started interaction: {instance_id}\n")
    
    # Simulate conversation
    # We'll use exact responses from the ground truth
    expected_responses = [
        MULTITURN_DIALOG_DATA["messages"][2]["content"],  # First assistant response
        MULTITURN_DIALOG_DATA["messages"][4]["content"],  # Second assistant response
    ]
    
    conversation_history = [
        MULTITURN_DIALOG_DATA["messages"][0],  # System message
    ]
    
    for turn_idx in range(len(expected_responses)):
        print(f"Turn {turn_idx + 1}:")
        print(f"  User: {MULTITURN_DIALOG_DATA['messages'][2*turn_idx + 1]['content'][:60]}...")
        
        # Add user message to conversation
        conversation_history.append(MULTITURN_DIALOG_DATA["messages"][2*turn_idx + 1])
        
        # Use the exact expected response
        generated_response = expected_responses[turn_idx]
        print(f"  Generated response: {generated_response[:80]}...")
        
        # Process the turn
        should_terminate, next_prompt, reward, metadata = await simulate_dialog_turn(
            interaction=interaction,
            instance_id=instance_id,
            conversation_history=conversation_history,
            generated_response=generated_response,
            turn_index=turn_idx,
        )
        
        print(f"  Reward: {reward}")
        print(f"  Should terminate: {should_terminate}")
        print(f"  Average reward so far: {metadata['average_reward']:.2f}")
        
        # Add generated response to history for next iteration
        conversation_history.append({"role": "assistant", "content": generated_response})
        
        if should_terminate:
            print(f"\n  ✓ Dialog completed!")
            break
        print()
    
    # Calculate final score
    final_score = await interaction.calculate_score(instance_id)
    print(f"\nFinal score: {final_score:.2f}")
    
    # Finalize
    await interaction.finalize_interaction(instance_id)
    print("Interaction finalized.\n")


async def demo_partial_match():
    """Demo: Generate responses with partial errors (rewards should be 0.0)."""
    print("\n" + "="*80)
    print("DEMO 2: Partial Match (Some rewards should be 0.0)")
    print("="*80 + "\n")
    
    # Initialize interaction
    interaction = MultiTurnDialogInteraction(config={"name": "multiturn_dialog"})
    
    # Start interaction
    instance_id = await interaction.start_interaction(
        messages=MULTITURN_DIALOG_DATA["messages"]
    )
    print(f"Started interaction: {instance_id}\n")
    
    # Simulate conversation with slightly modified responses
    generated_responses = [
        # First turn: slightly different from expected
        "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n-1)\n\nThis calculates factorial recursively.",
        # Second turn: exact match
        MULTITURN_DIALOG_DATA["messages"][4]["content"],
    ]
    
    conversation_history = [
        MULTITURN_DIALOG_DATA["messages"][0],  # System message
    ]
    
    for turn_idx in range(len(generated_responses)):
        print(f"Turn {turn_idx + 1}:")
        user_msg = MULTITURN_DIALOG_DATA["messages"][2*turn_idx + 1]["content"]
        print(f"  User: {user_msg[:60]}...")
        
        # Add user message to conversation
        conversation_history.append(MULTITURN_DIALOG_DATA["messages"][2*turn_idx + 1])
        
        generated_response = generated_responses[turn_idx]
        print(f"  Generated response: {generated_response[:80]}...")
        
        # Process the turn
        should_terminate, next_prompt, reward, metadata = await simulate_dialog_turn(
            interaction=interaction,
            instance_id=instance_id,
            conversation_history=conversation_history,
            generated_response=generated_response,
            turn_index=turn_idx,
        )
        
        print(f"  Reward: {reward}")
        print(f"  Should terminate: {should_terminate}")
        print(f"  Average reward so far: {metadata['average_reward']:.2f}")
        
        # Add generated response to history
        conversation_history.append({"role": "assistant", "content": generated_response})
        
        if should_terminate:
            print(f"\n  ✓ Dialog completed!")
            break
        print()
    
    # Calculate final score
    final_score = await interaction.calculate_score(instance_id)
    print(f"\nFinal score: {final_score:.2f}")
    
    # Finalize
    await interaction.finalize_interaction(instance_id)
    print("Interaction finalized.\n")


async def demo_integration_with_sglang():
    """
    Demo: How to integrate with SGLang rollout.
    
    This shows the recommended flow for using MultiTurnDialogInteraction with SGLang.
    In actual training, this would be called from the rollout worker.
    """
    print("\n" + "="*80)
    print("DEMO 3: Integration Pattern with SGLang Rollout")
    print("="*80 + "\n")
    
    print("""
Configuration for YAML:

actor_rollout_ref:
  rollout_mode: sglang  # Use SGLang for multi-turn support
  rollout:
    mode: async
    server_mode: true
    tensor_model_parallel_size: 1
    enable_multiturn: true  # Enable multi-turn support
    multiturn:
      enable: true
      interaction_cls: verl.interactions.multiturn_dialog_interaction.MultiTurnDialogInteraction
    
data:
  train_files: ["data/multiturn_dialog_data.jsonl"]
  val_files: ["data/multiturn_dialog_val.jsonl"]
  data_loader:
    type: multiturn_dialog
    # Each item should contain:
    # {
    #   "data_source": "multiturn_dialog",
    #   "messages": [
    #       {"role": "system", "content": "..."},
    #       {"role": "user", "content": "Q1"},
    #       {"role": "assistant", "content": "Expected A1"},
    #       {"role": "user", "content": "Q2"},
    #       {"role": "assistant", "content": "Expected A2"},
    #   ]
    # }

Data Format:
- Each conversation item contains a complete multi-turn dialog
- Messages are in standard OpenAI format: [system, user1, assistant1, user2, assistant2, ...]
- The model will generate one assistant response per turn
- Rewards are computed by comparing against the expected assistant responses
- Reward: 1.0 for exact match, 0.0 for any mismatch

Training Flow:
1. Rollout worker loads conversation data
2. For each conversation:
   a. Initialize MultiTurnDialogInteraction with start_interaction()
   b. For each turn:
      - Model generates assistant response (via SGLang)
      - Call generate_response() to get reward
      - If not terminated, continue with next turn prompt
   c. Finalize with finalize_interaction()
3. Rewards are collected and used for PPO training
    """)
    
    print("Example data loading code:\n")
    print("""
# Load multi-turn dialog data
conversations = []
with open('multiturn_dialog_data.jsonl') as f:
    for line in f:
        conversations.append(json.loads(line))

# Process each conversation
for conv_data in conversations:
    # Initialize interaction
    interaction = MultiTurnDialogInteraction(config={})
    instance_id = await interaction.start_interaction(
        messages=conv_data["messages"]
    )
    
    # Rollout turns
    current_messages = [conv_data["messages"][0]]  # Start with system message
    for turn_idx in range(num_turns):
        # Add user message
        current_messages.append(conv_data["messages"][2*turn_idx + 1])
        
        # Generate assistant response via SGLang
        response = await sglang_client.generate(current_messages)
        
        # Evaluate response
        should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
            instance_id=instance_id,
            messages=current_messages + [{"role": "assistant", "content": response}]
        )
        
        # Collect rewards
        rewards.append(reward)
        
        if should_terminate:
            break
    
    # Cleanup
    await interaction.finalize_interaction(instance_id)
    """)


async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("Multi-Turn Dialog Interaction Demo")
    print("="*80)
    
    # Run demos
    await demo_exact_match()
    await demo_partial_match()
    await demo_integration_with_sglang()
    
    print("\n" + "="*80)
    print("All demos completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

