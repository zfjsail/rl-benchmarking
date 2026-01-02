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

import logging
import os
from typing import Any, Optional, List, Dict, Tuple, Literal
from uuid import uuid4

from verl.utils.reward_score import gsm8k

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class NDInteraction(BaseInteraction):
    """Interaction for QAQAQA format tasks.
    
    QAQAQA format: [Q1, A1_gt, Q2, A2_gt, Q3, A3_gt, ...]
    
    Flow:
    1. Initial prompt: [系统消息, Q1, A1_gt, Q2, A2_gt, ..., Qn, An_gt]
    2. Agent generates: A1_rollout (replaces A1_gt position during rollout)
    3. Interaction evaluates: A1_rollout against A1_gt, returns reward + next question Q2
    4. Agent generates: A2_rollout 
    5. Interaction evaluates: A2_rollout against A2_gt, returns reward + next question Q3
    ... (repeat)
    
    Reward System:
    - Step Reward: Calculated during each step (per-step or batch at end)
    - Final Reward: Calculated once at the end of interaction
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        qaqaqa_pairs: Optional[List[Dict]] = None,
        step_reward_mode: Literal["per_step", "batch_at_end"] = "per_step",
        **kwargs
    ) -> str:
        """Start a new QAQAQA interaction instance.
        
        Args:
            instance_id: The instance ID
            qaqaqa_pairs: List of {question, ground_truth_answer} dicts
                         Format: [
                            {"question": "Q1", "ground_truth": "A1_gt"},
                            {"question": "Q2", "ground_truth": "A2_gt"},
                            ...
                         ]
            step_reward_mode: "per_step" - return reward immediately after each step
                             "batch_at_end" - return 0 for intermediate steps, compute all rewards at end
            **kwargs: Additional arguments for the task
        
        Returns:
            The instance ID
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        if qaqaqa_pairs is None:
            qaqaqa_pairs = []
        
        # Extract questions and ground truths from pairs
        questions = [pair.get("question", "") for pair in qaqaqa_pairs]
        ground_truths = [pair.get("ground_truth", "") for pair in qaqaqa_pairs]
        
        self._instance_dict[instance_id] = {
            "qaqaqa_pairs": qaqaqa_pairs,
            "questions": questions,
            "ground_truths": ground_truths,
            "current_turn": 0,
            "generated_answers": [],
            "step_rewards": [],
            "step_reward_mode": step_reward_mode,
            "turn_history": [],
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Generate response and evaluate for the current question in QAQAQA format.
        
        Args:
            instance_id: The instance ID
            messages: The conversation messages
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (should_terminate, next_question_or_feedback, reward, metadata_dict)
        """
        instance = self._instance_dict[instance_id]
        current_turn = instance["current_turn"]
        
        # Extract the latest assistant response
        generated_answer = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                generated_answer = item.get("content", "")
                break

        ground_truth = instance["ground_truths"][current_turn]
        current_question = instance["questions"][current_turn]
        
        instance["generated_answers"].append(generated_answer)
        
        # Calculate step reward for current answer
        step_reward = await self._evaluate_answer(
            generated_answer,
            ground_truth,
        )
        instance["step_rewards"].append(step_reward)
        
        # Store turn information
        instance["turn_history"].append({
            "turn": current_turn,
            "question": current_question,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "step_reward": step_reward,
        })
        
        # Move to next turn
        instance["current_turn"] += 1
        
        # Check if interaction should terminate
        should_terminate = instance["current_turn"] >= len(instance["questions"])
        
        # Determine reward to return based on mode
        if instance["step_reward_mode"] == "per_step":
            # Return step reward immediately
            current_reward = step_reward
        else:  # batch_at_end
            # Return 0 for intermediate steps, rewards only at end
            current_reward = 0.0
            if should_terminate:
                # At termination, return average of all step rewards
                current_reward = sum(instance["step_rewards"]) / len(instance["step_rewards"]) if instance["step_rewards"] else 0.0
        
        # Calculate final reward at termination
        final_reward = 0.0
        if should_terminate:
            final_reward = await self._calculate_final_reward(
                instance["generated_answers"],
                instance["ground_truths"]
            )
        
        # Prepare response
        if should_terminate:
            response = "All questions have been answered. Interaction complete."
        else:
            next_question_text = instance["questions"][instance["current_turn"]]
            next_ground_truth = instance["ground_truths"][instance["current_turn"]]
            response = f"{next_question_text}\n{next_ground_truth}"
        
        # Calculate statistics
        avg_step_reward = sum(instance["step_rewards"]) / len(instance["step_rewards"]) if instance["step_rewards"] else 0.0
        
        final_metadata = {
            "turn": current_turn,
            "turns_completed": instance["current_turn"],
            "total_turns": len(instance["ground_truths"]),
            "qa_history": instance["turn_history"],
            "all_step_rewards": instance["step_rewards"],
            "step_reward": step_reward,
            "average_step_reward": avg_step_reward,
            "final_reward": final_reward if should_terminate else None,
            "should_terminate": should_terminate,
            "step_reward_mode": instance["step_reward_mode"],
        }
        
        return should_terminate, response, current_reward, final_metadata

    async def _evaluate_answer(
        self,
        response: str,
        ground_truth: str,
    ) -> float:
        """Evaluate a single answer against ground truth.
        
        Args:
            response: The generated response
            ground_truth: The correct answer
        
        Returns:
            Step reward score (0.0 to 1.0)
        """
        if response == ground_truth:
            return 1.0
        else:
            return 0.0

    async def _calculate_final_reward(
        self,
        generated_answers: List[str],
        ground_truths: List[str]
    ) -> float:
        """Calculate final reward based on all generated answers and ground truths.
        
        Currently returns average of step rewards. Can be extended for more complex
        final reward calculations.
        
        Args:
            generated_answers: All generated answers for the interaction
            ground_truths: All ground truth answers
        
        Returns:
            Final reward score (0.0 to 1.0)
        """
        # Default final reward: average of step-by-step correctness
        if not generated_answers:
            return 0.0
        
        correct_count = sum(1 for gen, gt in zip(generated_answers, ground_truths) if gen == gt)
        return correct_count / len(generated_answers)

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate the overall score for the interaction.
        
        Returns:
            The average of all step rewards
        """
        instance = self._instance_dict[instance_id]
        
        if not instance["step_rewards"]:
            return 0.0

        return sum(instance["step_rewards"]) / len(instance["step_rewards"])

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Finalize the interaction and clean up resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]