# QAQAQA 交互 Logits 集成实现示例

"""
这个文件展示了一个完整的、可直接使用的 QAQAQA 交互扩展，
集成了生成序列的 logits 跟踪和 final reward 计算。

不需要修改 VERL 的核心代码，只需创建这个新的交互类。

使用方法：
1. 将此文件放在 verl/interactions/ 目录下
2. 在配置中注册此交互类
3. 在 Rollout 过程中调用相关方法存储 logits
"""

import asyncio
import logging
import os
from typing import Any, Optional, List, Dict, Tuple
from uuid import uuid4
from dataclasses import dataclass, field
import numpy as np

import torch

from verl.interactions.base import BaseInteraction
from verl.utils.reward_score import gsm8k

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class TurnLogits:
    """单轮生成的 logits 信息"""
    token_ids: np.ndarray  # (seq_len,) 生成的 token IDs
    log_probs: np.ndarray  # (seq_len,) 每个 token 的 log probability
    avg_log_prob: float = 0.0  # 平均 log probability
    min_log_prob: float = 0.0  # 最小 log probability  
    max_log_prob: float = 0.0  # 最大 log probability
    total_log_prob: float = 0.0  # 累计 log probability（用于计算整体得分）


class QAQAQAInteractionWithLogitsExtension(BaseInteraction):
    """
    QAQAQA 交互的扩展版本，集成了生成序列的 logits 跟踪。
    
    这个类继承自 BaseInteraction，添加了以下功能：
    1. 存储和检索每一轮的 logits 信息
    2. 计算基于 logits 的奖励
    3. 提供最终奖励计算（考虑 logits 和 step rewards）
    
    主要改动：
    - 添加 `_instance_logits` 用于存储 logits 信息
    - 提供 `register_turn_logits()` 方法在 rollout 后调用
    - 提供 `compute_final_reward()` 方法进行最终计算
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self._instance_logits = {}  # 新增：存储 logits
    
    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        qaqaqa_pairs: Optional[List[Dict]] = None,
        evaluation_method: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        初始化 QAQAQA 交互实例。
        
        与基础版本相同，但添加了 logits 存储初始化。
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        if qaqaqa_pairs is None:
            qaqaqa_pairs = []
        
        # 提取 Q 和 A
        questions = [pair.get("question", "") for pair in qaqaqa_pairs]
        ground_truths = [pair.get("ground_truth", "") for pair in qaqaqa_pairs]
        
        self._instance_dict[instance_id] = {
            "qaqaqa_pairs": qaqaqa_pairs,
            "questions": questions,
            "ground_truths": ground_truths,
            "current_turn": 0,
            "generated_answers": [],
            "step_rewards": [],
            "evaluation_method": evaluation_method or "strict",
            "turn_history": [],
        }
        
        # 新增：初始化 logits 存储
        self._instance_logits[instance_id] = {}
        
        return instance_id
    
    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> tuple[bool, str, float, dict]:
        """
        与基础版本相同的生成响应逻辑。
        """
        instance = self._instance_dict[instance_id]
        current_turn = instance["current_turn"]
        
        # 提取最后一个 assistant 响应
        generated_answer = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                generated_answer = item.get("content", "")
                break
        
        # 检查是否还有问题
        if current_turn >= len(instance["ground_truths"]):
            reward = 0.0
            instance["generated_answers"].append(generated_answer)
            instance["step_rewards"].append(reward)
            should_terminate = True
            response = "All questions have been answered."
            
            avg_reward = sum(instance["step_rewards"]) / len(instance["step_rewards"]) if instance["step_rewards"] else 0.0
            final_metadata = {
                "turn": current_turn,
                "turns_completed": len(instance["step_rewards"]),
                "total_turns": len(instance["ground_truths"]),
                "all_rewards": instance["step_rewards"],
                "average_reward": avg_reward,
                "qa_history": instance["turn_history"],
                "should_terminate": True,
            }
            return should_terminate, response, reward, final_metadata
        
        # 评估答案
        ground_truth = instance["ground_truths"][current_turn]
        current_question = instance["questions"][current_turn]
        
        instance["generated_answers"].append(generated_answer)
        
        reward = await self._evaluate_answer(
            generated_answer,
            ground_truth,
            instance["evaluation_method"]
        )
        instance["step_rewards"].append(reward)
        
        # 记录 turn 历史
        instance["turn_history"].append({
            "turn": current_turn,
            "question": current_question,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "reward": reward,
        })
        
        instance["current_turn"] += 1
        
        should_terminate = False
        if instance["current_turn"] < len(instance["questions"]):
            next_question_text = instance["questions"][instance["current_turn"]]
            next_ground_truth = instance["ground_truths"][instance["current_turn"]]
            response = f"{next_question_text}\n{next_ground_truth}"
        else:
            should_terminate = True
            response = "All questions have been answered. Interaction complete."
        
        avg_reward = sum(instance["step_rewards"]) / len(instance["step_rewards"]) if instance["step_rewards"] else 0.0
        
        final_metadata = {
            "turn": current_turn,
            "turns_completed": instance["current_turn"],
            "total_turns": len(instance["ground_truths"]),
            "qa_history": instance["turn_history"],
            "all_rewards": instance["step_rewards"],
            "current_reward": reward,
            "average_reward": avg_reward,
            "should_terminate": should_terminate,
        }
        
        return should_terminate, response, reward, final_metadata
    
    # ========================================================================
    # 新增方法：Logits 相关功能
    # ========================================================================
    
    async def register_turn_logits(
        self,
        instance_id: str,
        turn_index: int,
        token_ids: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        """
        在 rollout 完成后注册某一轮的 logits 信息。
        
        这个方法应该在 SGLang rollout 完成后调用，
        用来存储每个生成的答案对应的 logits 信息。
        
        Args:
            instance_id: 交互实例 ID
            turn_index: 当前轮次索引
            token_ids: 生成的 token IDs，形状 (seq_len,) 或 (1, seq_len)
            log_probs: 每个 token 的 log probability，形状 (seq_len,) 或 (1, seq_len)
        
        Example:
            # 在 rollout 引擎完成生成后调用
            await interaction.register_turn_logits(
                instance_id="xyz-123",
                turn_index=0,
                token_ids=torch.tensor([2392, 414, 100]),
                log_probs=torch.tensor([-0.5, -0.8, -0.3]),
            )
        """
        # 确保张量是 1D 的
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze()
        if log_probs.dim() > 1:
            log_probs = log_probs.squeeze()
        
        # 转换为 numpy 以便存储
        token_ids_np = token_ids.cpu().numpy() if isinstance(token_ids, torch.Tensor) else token_ids
        log_probs_np = log_probs.cpu().numpy() if isinstance(log_probs, torch.Tensor) else log_probs
        
        # 计算统计信息
        avg_log_prob = float(log_probs_np.mean())
        min_log_prob = float(log_probs_np.min())
        max_log_prob = float(log_probs_np.max())
        total_log_prob = float(log_probs_np.sum())
        
        # 存储
        if instance_id not in self._instance_logits:
            self._instance_logits[instance_id] = {}
        
        self._instance_logits[instance_id][turn_index] = TurnLogits(
            token_ids=token_ids_np,
            log_probs=log_probs_np,
            avg_log_prob=avg_log_prob,
            min_log_prob=min_log_prob,
            max_log_prob=max_log_prob,
            total_log_prob=total_log_prob,
        )
        
        logger.debug(f"Registered logits for instance {instance_id}, turn {turn_index}: "
                    f"avg_log_prob={avg_log_prob:.4f}, total_tokens={len(log_probs_np)}")
    
    async def get_turn_logits(
        self,
        instance_id: str,
        turn_index: int
    ) -> Optional[TurnLogits]:
        """
        获取某一轮的 logits 信息。
        
        Returns:
            TurnLogits 对象或 None
        """
        return self._instance_logits.get(instance_id, {}).get(turn_index)
    
    async def get_all_logits(
        self,
        instance_id: str
    ) -> Dict[int, TurnLogits]:
        """
        获取某个 instance 的所有轮次的 logits。
        
        Returns:
            {turn_index: TurnLogits} 字典
        """
        return self._instance_logits.get(instance_id, {})
    
    async def compute_final_reward_with_logits(
        self,
        instance_id: str,
        logit_weight: float = 0.1,
    ) -> Dict[str, Any]:
        """
        基于存储的 logits 和 step rewards 计算最终奖励。
        
        Args:
            instance_id: 交互实例 ID
            logit_weight: logits 对最终奖励的权重，范围 [0, 1]
                         0: 完全忽略 logits
                         1: 完全基于 logits
                         0.1: 10% 基于 logits，90% 基于 step rewards（推荐）
        
        Returns:
            {
                "final_reward": float,  # 最终奖励
                "step_rewards": List[float],  # 原始 step rewards
                "logits_summary": {
                    "avg_log_prob_per_turn": List[float],
                    "avg_log_prob_overall": float,
                    "total_tokens": int,
                    "consistency_score": float,
                },
                "details": {...}
            }
        """
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            raise ValueError(f"Instance {instance_id} not found")
        
        step_rewards = instance["step_rewards"]
        all_logits = await self.get_all_logits(instance_id)
        
        # 基础奖励
        base_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        
        if not all_logits:
            # 没有 logits，返回基础奖励
            logger.warning(f"No logits available for instance {instance_id}, using base reward")
            return {
                "final_reward": base_reward,
                "step_rewards": step_rewards,
                "logits_summary": None,
                "details": {
                    "num_turns": len(step_rewards),
                    "warning": "No logits registered"
                }
            }
        
        # 收集 logits 统计
        avg_log_probs_per_turn = []
        total_tokens = 0
        
        for turn_idx in sorted(all_logits.keys()):
            turn_logits = all_logits[turn_idx]
            avg_log_probs_per_turn.append(turn_logits.avg_log_prob)
            total_tokens += len(turn_logits.log_probs)
        
        # 计算总体平均 log prob
        avg_log_prob_overall = np.mean(avg_log_probs_per_turn) if avg_log_probs_per_turn else 0.0
        
        # 计算一致性分数（方差越小，一致性越高）
        # 范围 [0, 1]，1 表示完全一致（所有轮都有相同的 log prob）
        if len(avg_log_probs_per_turn) > 1:
            variance = np.var(avg_log_probs_per_turn)
            max_possible_variance = variance if variance > 0 else 1.0
            consistency_score = 1.0 - min(abs(variance / max_possible_variance), 1.0)
        else:
            consistency_score = 1.0
        
        # 计算最终奖励
        # 方法 1: 直接加权
        # final_reward = (1 - logit_weight) * base_reward + logit_weight * exp(avg_log_prob)
        
        # 方法 2: 使用 likelihood ratio（推荐）
        # 将 log prob 转换为 likelihood
        likelihood_ratio = np.exp(avg_log_prob_overall)
        final_reward = (1 - logit_weight) * base_reward + logit_weight * likelihood_ratio
        
        return {
            "final_reward": float(final_reward),
            "base_reward": float(base_reward),
            "step_rewards": step_rewards,
            "logits_summary": {
                "avg_log_prob_per_turn": avg_log_probs_per_turn,
                "avg_log_prob_overall": float(avg_log_prob_overall),
                "total_tokens": total_tokens,
                "consistency_score": float(consistency_score),
            },
            "details": {
                "num_turns": len(step_rewards),
                "turns_with_logits": len(all_logits),
                "logit_weight": logit_weight,
                "likelihood_ratio": float(likelihood_ratio),
            }
        }
    
    async def _evaluate_answer(
        self,
        response: str,
        ground_truth: str,
        evaluation_method: str = "strict"
    ) -> float:
        """评估单个答案（与基础版本相同）"""
        if evaluation_method == "strict":
            return gsm8k.compute_score(
                response,
                ground_truth,
                method="strict",
                format_score=0.0,
                score=1.0,
            )
        elif evaluation_method == "flexible":
            return gsm8k.compute_score(
                response,
                ground_truth,
                method="flexible",
                format_score=0.0,
                score=1.0,
            )
        else:
            return gsm8k.compute_score(
                response,
                ground_truth,
                method="strict",
                format_score=0.0,
                score=1.0,
            )
    
    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """计算总体得分"""
        instance = self._instance_dict[instance_id]
        if not instance["step_rewards"]:
            return 0.0
        return sum(instance["step_rewards"]) / len(instance["step_rewards"])
    
    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """清理资源"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
        if instance_id in self._instance_logits:
            del self._instance_logits[instance_id]


# ============================================================================
# 使用示例
# ============================================================================

async def example_usage():
    """
    展示如何使用带有 logits 的 QAQAQA 交互。
    """
    # 创建交互实例
    interaction = QAQAQAInteractionWithLogitsExtension({})
    
    # 定义问题和标准答案
    qaqaqa_pairs = [
        {"question": "What is 2+2?", "ground_truth": "4"},
        {"question": "What is 2+3?", "ground_truth": "5"},
    ]
    
    # 初始化交互
    instance_id = await interaction.start_interaction(
        qaqaqa_pairs=qaqaqa_pairs,
        evaluation_method="strict"
    )
    print(f"Started interaction: {instance_id}")
    
    # 模拟 Turn 0
    messages_turn0 = [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]
    
    should_terminate, response, reward, metadata = \
        await interaction.generate_response(instance_id, messages_turn0)
    
    print(f"Turn 0 - Reward: {reward}, Terminate: {should_terminate}")
    
    # 注册 Turn 0 的 logits（这会在 rollout 引擎中调用）
    await interaction.register_turn_logits(
        instance_id=instance_id,
        turn_index=0,
        token_ids=torch.tensor([2392, 414]),
        log_probs=torch.tensor([-0.5, -0.3]),
    )
    
    # 模拟 Turn 1
    messages_turn1 = [
        {"role": "system", "content": "You are a math expert."},
        {"role": "user", "content": "What is 2+3?"},
        {"role": "assistant", "content": "The answer is 5."}
    ]
    
    should_terminate, response, reward, metadata = \
        await interaction.generate_response(instance_id, messages_turn1)
    
    print(f"Turn 1 - Reward: {reward}, Terminate: {should_terminate}")
    
    # 注册 Turn 1 的 logits
    await interaction.register_turn_logits(
        instance_id=instance_id,
        turn_index=1,
        token_ids=torch.tensor([2392, 414]),
        log_probs=torch.tensor([-0.4, -0.35]),
    )
    
    # 计算最终奖励（考虑 logits）
    final_result = await interaction.compute_final_reward_with_logits(
        instance_id=instance_id,
        logit_weight=0.1  # 10% 权重给 logits
    )
    
    print("\n=== Final Result ===")
    print(f"Final Reward: {final_result['final_reward']:.4f}")
    print(f"Base Reward: {final_result['base_reward']:.4f}")
    print(f"Logits Summary:")
    print(f"  - Avg log prob per turn: {final_result['logits_summary']['avg_log_prob_per_turn']}")
    print(f"  - Avg log prob overall: {final_result['logits_summary']['avg_log_prob_overall']:.4f}")
    print(f"  - Consistency score: {final_result['logits_summary']['consistency_score']:.4f}")
    
    # 清理
    await interaction.finalize_interaction(instance_id)


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())
