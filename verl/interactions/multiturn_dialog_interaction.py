# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Multi-turn Dialog Interaction Module

This module provides an interaction implementation for multi-turn dialog tasks.
Each turn consists of:
- User message (from dataset)
- Expected assistant response (ground truth)
- Generated assistant response (from model rollout)

Reward is calculated as exact string matching (1.0 if match, 0.0 otherwise).

Example data format:
{
    "data_source": "multiturn_dialog",
    "reward_model": {"style": "rule"},
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Expected Answer 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Expected Answer 2"},
    ]
}
"""

import logging
import os
from typing import Any, Optional, List, Dict
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MultiTurnDialogInteraction(BaseInteraction):
    """Interaction for multi-turn dialog tasks.
    
    Supports multi-turn conversations where:
    - Each turn has a user message and an expected assistant response
    - Agent generates responses one turn at a time
    - Rewards are based on exact string matching between generated and expected responses
    - Reward = 1.0 if exact match, 0.0 otherwise
    
    Data Format:
    {
        "data_source": "multiturn_dialog",
        "reward_model": {"style": "rule"},
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Expected answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Expected answer 2"},
            ...
        ]
    }
    
    Flow:
    1. start_interaction: Parse messages into turns (pairs of user-assistant messages)
    2. generate_response: Receive generated assistant response, evaluate against ground truth
    3. Return: (should_terminate, next_prompt_or_feedback, reward, metadata)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Start a new multi-turn dialog interaction instance.
        
        This method parses the input messages to extract dialog turns, where each turn
        consists of a user message and an expected assistant response.
        
        Args:
            instance_id: The instance ID. If None, a UUID will be generated.
            messages: Complete conversation including system message, user prompts, and expected assistant responses.
                     Format: [
                        {"role": "system", "content": "System instruction"},
                        {"role": "user", "content": "Question 1"},
                        {"role": "assistant", "content": "Expected Answer 1"},
                        {"role": "user", "content": "Question 2"},
                        {"role": "assistant", "content": "Expected Answer 2"},
                        ...
                     ]
            **kwargs: Additional arguments for future extensibility
        
        Returns:
            The instance ID (either provided or newly generated)
            
        Raises:
            ValueError: If messages contain invalid format or no valid turns found
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        if messages is None:
            messages = []
        
        # Parse messages to extract dialog turns
        # Structure: [system_msg], then alternating user/assistant pairs
        system_msg = None
        dialog_turns = []
        unpaired_user_count = 0
        
        for msg in messages:
            if not isinstance(msg, dict):
                logger.warning(f"Skipping non-dict message in instance {instance_id}")
                continue
                
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_msg = content
            elif role == "user":
                # Start a new turn with user message
                dialog_turns.append({
                    "user": content,
                    "assistant_expected": None,
                    "assistant_generated": None,
                })
                unpaired_user_count += 1
            elif role == "assistant" and dialog_turns:
                # Fill in the expected assistant response for the last turn
                if dialog_turns[-1]["assistant_expected"] is None:
                    dialog_turns[-1]["assistant_expected"] = content
                    unpaired_user_count -= 1
                else:
                    logger.warning(f"Found consecutive assistant messages in instance {instance_id}, skipping")
        
        # Validate that we have complete pairs (user + expected assistant)
        valid_turns = [
            turn for turn in dialog_turns 
            if turn["assistant_expected"] is not None
        ]
        
        # Log warnings if there are unpaired turns
        if unpaired_user_count > 0:
            logger.warning(
                f"Instance {instance_id}: Found {unpaired_user_count} unpaired user message(s) "
                f"(no expected assistant response). They will be ignored."
            )
        
        if not valid_turns:
            logger.warning(f"Instance {instance_id}: No valid dialog turns found in messages")
        
        self._instance_dict[instance_id] = {
            "system_message": system_msg,
            "all_messages": messages,
            "dialog_turns": valid_turns,
            "current_turn": 0,
            "turn_rewards": [],
            "turn_history": [],
        }
        
        logger.info(
            f"Started interaction {instance_id} with {len(valid_turns)} valid turns "
            f"(total messages: {len(messages)})"
        )
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Process generated response and return evaluation.
        
        This method:
        1. Extracts the latest assistant response from messages
        2. Compares it with the ground truth for the current turn
        3. Calculates reward (1.0 for exact match, 0.0 otherwise)
        4. Advances to the next turn or terminates
        5. Returns evaluation results
        
        Args:
            instance_id: The instance ID
            messages: Complete conversation history from rollout (including system, all user messages, 
                     and the latest generated assistant response)
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (should_terminate, next_prompt_or_feedback, reward, metadata)
                - should_terminate: True if all turns are completed
                - next_prompt_or_feedback: If continuing, contains next user prompt. 
                                          If terminating, contains completion message
                - reward: 1.0 if current response exactly matches ground truth, 0.0 otherwise
                - metadata: Dictionary with turn statistics
        """
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")
        
        instance = self._instance_dict[instance_id]
        current_turn_idx = instance["current_turn"]
        
        # Extract the latest assistant response from messages
        generated_response = ""
        if messages:
            for i in range(len(messages) - 1, -1, -1):
                item = messages[i]
                if isinstance(item, dict) and item.get("role") == "assistant":
                    generated_response = item.get("content", "")
                    break
        
        # Get ground truth for current turn
        if current_turn_idx >= len(instance["dialog_turns"]):
            should_terminate = True
            next_prompt = "All dialog turns completed."
            reward = 0.0
            metadata = {
                "error": "Already completed all turns",
                "current_turn": current_turn_idx,
                "total_turns": len(instance["dialog_turns"]),
            }
            logger.warning(f"Instance {instance_id}: Attempted to generate response beyond all turns")
            return should_terminate, next_prompt, reward, metadata
        
        current_turn = instance["dialog_turns"][current_turn_idx]
        expected_response = current_turn["assistant_expected"]
        
        # Calculate reward based on Yes/No matching
        def extract_yes_no(text):
            """提取文本中的Yes或No（整词匹配），返回 'Yes', 'No' 或 None"""
            import re
            text_upper = text.upper()
            # 使用正则表达式匹配完整的单词YES或NO，不匹配作为其他单词一部分的情况
            yes_matches = len(re.findall(r'\bYES\b', text_upper))
            no_matches = len(re.findall(r'\bNO\b', text_upper))
            
            # 如果两者都出现或都没出现，返回None
            if (yes_matches > 0 and no_matches > 0) or (yes_matches == 0 and no_matches == 0):
                return None
            
            return "Yes" if yes_matches > 0 else "No"
        
        generated_answer = extract_yes_no(generated_response)
        expected_answer = extract_yes_no(expected_response)
        
        # 如果两个都提取成功且相同，则reward=1.0；否则0.0
        reward = 1.0 if generated_answer is not None and generated_answer == expected_answer else 0.0
        
        # Store generated response and reward
        current_turn["assistant_generated"] = generated_response
        instance["turn_rewards"].append(reward)
        
        # Record turn history for analysis
        instance["turn_history"].append({
            "turn_index": current_turn_idx,
            "user_message": current_turn["user"],
            "expected_response": expected_response,
            "generated_response": generated_response,
            "match": reward == 1.0,
            "reward": reward,
        })
        
        # 记录到本地res.txt文件用于查看训练过程的输入输出
        with open("./res.txt", "a", encoding="utf-8") as f:
            f.write(f"Turn {current_turn_idx}: Expected={expected_response} | Generated={generated_response} | Reward={reward}\n")
        
        # Move to next turn
        instance["current_turn"] += 1
        
        # Determine if interaction should terminate
        should_terminate = instance["current_turn"] >= len(instance["dialog_turns"])
        
        # Prepare next prompt or completion message
        if should_terminate:
            next_prompt = "Dialog completed. All turns have been processed."
            logger.info(f"Instance {instance_id}: Dialog completed after {instance['current_turn']} turns")
        else:
            # Prepare next user prompt for the next turn
            next_turn = instance["dialog_turns"][instance["current_turn"]]
            next_prompt = next_turn["user"]
        
        # Calculate statistics
        avg_reward = sum(instance["turn_rewards"]) / len(instance["turn_rewards"]) if instance["turn_rewards"] else 0.0
        num_matches = sum(1 for r in instance["turn_rewards"] if r > 0.5)
        accuracy = num_matches / len(instance["turn_rewards"]) if instance["turn_rewards"] else 0.0
        
        metadata = {
            "turn_index": current_turn_idx,
            "turns_completed": instance["current_turn"],
            "total_turns": len(instance["dialog_turns"]),
            "current_turn_reward": reward,
            "all_turn_rewards": instance["turn_rewards"],
            "num_matches": num_matches,
            "average_reward": avg_reward,
            "accuracy": accuracy,  # Proportion of turns with perfect match
            "should_terminate": should_terminate,
            "turn_history": instance["turn_history"],
        }
        
        return should_terminate, next_prompt, reward, metadata

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate the overall score for the interaction.
        
        The score is the average reward across all turns.
        - 1.0 = All turns matched exactly
        - 0.0 = No turns matched
        - 0.5 = Half of turns matched
        
        Args:
            instance_id: The instance ID
        
        Returns:
            Average reward across all turns (0.0 to 1.0)
            
        Raises:
            ValueError: If instance not found
        """
        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found")
        
        instance = self._instance_dict[instance_id]
        
        if not instance["turn_rewards"]:
            logger.warning(f"Instance {instance_id}: No rewards calculated yet")
            return 0.0
        
        avg_score = sum(instance["turn_rewards"]) / len(instance["turn_rewards"])
        logger.debug(f"Instance {instance_id}: Calculated score = {avg_score:.4f}")
        return avg_score

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Finalize the interaction and clean up resources.
        
        This method should be called after the dialog interaction is complete to release
        all associated state and resources.
        
        Args:
            instance_id: The instance ID
            **kwargs: Additional arguments for future extensibility
        """
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            final_score = await self.calculate_score(instance_id)
            completed_turns = instance["current_turn"]
            total_turns = len(instance["dialog_turns"])
            
            logger.info(
                f"Finalizing interaction {instance_id}: "
                f"completed {completed_turns}/{total_turns} turns, "
                f"final score: {final_score:.4f}"
            )
            
            del self._instance_dict[instance_id]
        else:
            logger.warning(f"Attempted to finalize non-existent instance {instance_id}")


async def example_usage():
    """Example of how to use MultiTurnDialogInteraction.
    
    This demonstrates the complete workflow for multi-turn dialog interaction.
    """
    import asyncio
    
    # Create interaction instance
    interaction = MultiTurnDialogInteraction({})
    
    # Example data with multiple turns
    # Format: system message followed by alternating user/assistant pairs
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Write a simple Python function to calculate factorial."
        },
        {
            "role": "assistant",
            "content": (
                "```python\n"
                "def factorial(n):\n"
                "    if n == 0 or n == 1:\n"
                "        return 1\n"
                "    else:\n"
                "        return n * factorial(n-1)\n"
                "```\n\n"
                "This is a recursive function to calculate the factorial of a number."
            )
        },
        {
            "role": "user",
            "content": "Can you make it iterative instead?"
        },
        {
            "role": "assistant",
            "content": (
                "```python\n"
                "def factorial(n):\n"
                "    result = 1\n"
                "    for i in range(1, n+1):\n"
                "        result *= i\n"
                "    return result\n"
                "```\n\n"
                "This is an iterative version of the factorial function."
            )
        },
    ]
    
    # Step 1: Start interaction
    print("=" * 70)
    print("Starting multi-turn dialog interaction...")
    print("=" * 70)
    instance_id = await interaction.start_interaction(messages=messages)
    print(f"Instance ID: {instance_id}\n")
    
    # Step 2: Simulate two turns of dialog
    # Turn 1: User asks for factorial function, model generates exact response
    print("Turn 1: Recursive factorial")
    print("-" * 70)
    turn1_messages = messages[:3]  # system + user + assistant (expected)
    
    # Simulate rollout: generate_response is called with model-generated response
    # In real scenario, this comes from SGLang rollout with generated content
    turn1_generated_messages = [
        messages[0],  # system
        messages[1],  # user
        {
            "role": "assistant",
            "content": (
                "```python\n"
                "def factorial(n):\n"
                "    if n == 0 or n == 1:\n"
                "        return 1\n"
                "    else:\n"
                "        return n * factorial(n-1)\n"
                "```\n\n"
                "This is a recursive function to calculate the factorial of a number."
            )
        }
    ]
    
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id,
        turn1_generated_messages
    )
    print(f"Reward: {reward} (Perfect Match: {reward == 1.0})")
    print(f"Should Terminate: {should_terminate}")
    print(f"Average Reward: {metadata['average_reward']:.4f}")
    print()
    
    # Turn 2: User asks to make it iterative, model generates exact response
    print("Turn 2: Iterative factorial")
    print("-" * 70)
    turn2_generated_messages = [
        messages[0],  # system
        messages[1],  # user
        messages[2],  # assistant (previous)
        messages[3],  # user (new)
        {
            "role": "assistant",
            "content": (
                "```python\n"
                "def factorial(n):\n"
                "    result = 1\n"
                "    for i in range(1, n+1):\n"
                "        result *= i\n"
                "    return result\n"
                "```\n\n"
                "This is an iterative version of the factorial function."
            )
        }
    ]
    
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id,
        turn2_generated_messages
    )
    print(f"Reward: {reward} (Perfect Match: {reward == 1.0})")
    print(f"Should Terminate: {should_terminate}")
    print(f"Average Reward: {metadata['average_reward']:.4f}")
    print(f"Accuracy: {metadata['accuracy']:.1%}")
    print()
    
    # Step 3: Calculate final score
    print("=" * 70)
    print("Interaction Results")
    print("=" * 70)
    final_score = await interaction.calculate_score(instance_id)
    print(f"Final Score: {final_score:.4f}")
    print(f"Turns Completed: {metadata['turns_completed']}/{metadata['total_turns']}")
    print()
    
    # Step 4: Finalize interaction
    await interaction.finalize_interaction(instance_id)
    print("Interaction finalized and resources cleaned up.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())

