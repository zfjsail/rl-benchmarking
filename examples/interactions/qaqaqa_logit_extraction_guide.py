# QAQAQA 交互中获取生成序列 Logits 的指南

"""
本文件说明如何在 QAQAQA 格式的多轮交互中获取和利用生成序列的 logits。

## 核心概念

### 1. VERL 中生成 logits 的来源

在 VERL 中，生成序列的 logits 可以通过以下方式获取：

**方式 A：从 SGLang 引擎直接获取（推荐用于 rollout）**
- SGLang 在生成时返回 `output_token_logprobs`
- 位置：在 `sglang_rollout.py` 中的 `_post_process_outputs()` 函数
- 返回内容：每个生成的 token 的 log probability 和 token ID

**方式 B：从模型的 logits 计算（推荐用于 reward 计算）**
- 使用模型的前向传播获取 logits
- 通过 `logprobs_from_logits()` 计算 log probabilities
- 位置：在 `verl/utils/torch_functional.py` 中
- 应用场景：策略梯度计算、KL 散度计算

### 2. QAQAQA 交互中的 logits 使用场景

在 QAQAQA 格式中，logits 的主要用途：

1. **计算 final reward 时需要**：
   - 每轮生成的答案的 log probability
   - 用于计算策略的 likelihood ratio

2. **获取时机**：
   - Rollout 阶段：SGLang 引擎生成时自动返回
   - Reward 计算阶段：重新计算或使用存储的 logits

3. **链接点**：
   - `AsyncRolloutRequest.rollout_log_probs`：存储生成的 log probs
   - `AsyncRolloutRequest.output_token_ids`：生成的 token IDs
"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.torch_functional import entropy_from_logits


# ============================================================================
# 方案 1: 在交互层中存储和获取 logits（轻量级方案）
# ============================================================================

@dataclass
class GenerationLogits:
    """存储生成序列的 logits 信息"""
    token_ids: torch.Tensor  # (seq_len,) 生成的 token IDs
    log_probs: torch.Tensor  # (seq_len,) 每个 token 的 log probability
    logits: Optional[torch.Tensor] = None  # (seq_len, vocab_size) 完整的 logits（可选，节省内存）


class QAQAQAInteractionWithLogits:
    """
    在 QAQAQA 交互中集成 logits 获取的扩展类
    
    这个类展示了如何在交互过程中跟踪和使用生成的 logits。
    """
    
    def __init__(self):
        self._instance_logits_cache = {}  # 存储每个 instance 的 logits
    
    async def store_generation_logits(
        self,
        instance_id: str,
        turn_index: int,
        token_ids: torch.Tensor,
        log_probs: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> None:
        """
        存储某一轮生成的 logits 信息。
        
        Args:
            instance_id: 交互实例 ID
            turn_index: 当前轮次索引
            token_ids: 生成的 token IDs，形状 (seq_len,)
            log_probs: 每个 token 的 log probability，形状 (seq_len,)
            logits: 可选的完整 logits，形状 (seq_len, vocab_size)
        
        Example:
            # 在 AsyncRolloutRequest 完成后调用此函数
            await interaction.store_generation_logits(
                instance_id="xyz",
                turn_index=0,
                token_ids=torch.tensor([2392, 414, 100]),
                log_probs=torch.tensor([-0.5, -0.8, -0.3]),
            )
        """
        if instance_id not in self._instance_logits_cache:
            self._instance_logits_cache[instance_id] = {}
        
        self._instance_logits_cache[instance_id][turn_index] = GenerationLogits(
            token_ids=token_ids,
            log_probs=log_probs,
            logits=logits,
        )
    
    async def get_turn_logits(
        self,
        instance_id: str,
        turn_index: int
    ) -> Optional[GenerationLogits]:
        """
        获取某一轮的 logits 信息。
        
        Args:
            instance_id: 交互实例 ID
            turn_index: 轮次索引
        
        Returns:
            GenerationLogits 对象或 None（如果不存在）
        """
        return self._instance_logits_cache.get(instance_id, {}).get(turn_index)
    
    async def get_all_logits(
        self,
        instance_id: str
    ) -> Dict[int, GenerationLogits]:
        """
        获取某个 instance 的所有轮次的 logits。
        
        Returns:
            {turn_index: GenerationLogits} 字典
        """
        return self._instance_logits_cache.get(instance_id, {})
    
    async def compute_final_reward_with_logits(
        self,
        instance_id: str,
        step_rewards: List[float],
        logits_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        基于 logits 和 step rewards 计算最终奖励。
        
        这是一个示例函数，展示如何结合 logits 信息计算最终奖励。
        
        Args:
            instance_id: 交互实例 ID
            step_rewards: 每一步的奖励值
            logits_weights: 可选的权重，用于加权平均 logits 的影响
        
        Returns:
            {
                "final_reward": float,  # 最终奖励
                "logit_weighted_reward": float,  # 考虑 logits 的奖励
                "avg_log_prob": float,  # 平均 log probability
                "details": {...}
            }
        
        Example:
            result = await interaction.compute_final_reward_with_logits(
                instance_id="xyz",
                step_rewards=[1.0, 0.8, 0.9],
                logits_weights=[0.3, 0.3, 0.4]
            )
        """
        all_logits = await self.get_all_logits(instance_id)
        
        if not all_logits:
            # 如果没有 logits 信息，返回简单的平均奖励
            avg_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
            return {
                "final_reward": avg_reward,
                "logit_weighted_reward": avg_reward,
                "avg_log_prob": 0.0,
                "details": {"warning": "No logits available"}
            }
        
        # 收集所有 log probabilities
        all_log_probs = []
        for turn_idx, gen_logits in all_logits.items():
            if gen_logits.log_probs is not None:
                all_log_probs.extend(gen_logits.log_probs.cpu().tolist())
        
        # 计算平均 log probability
        avg_log_prob = sum(all_log_probs) / len(all_log_probs) if all_log_probs else 0.0
        
        # 基础奖励
        avg_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        
        # 使用 logits 权重加权计算
        if logits_weights is None:
            logits_weights = [1.0 / len(step_rewards)] * len(step_rewards)
        
        logit_weighted_reward = sum(
            reward * weight * (1.0 + avg_log_prob)
            for reward, weight in zip(step_rewards, logits_weights)
        )
        
        return {
            "final_reward": avg_reward,
            "logit_weighted_reward": logit_weighted_reward,
            "avg_log_prob": avg_log_prob,
            "token_log_probs": all_log_probs,
            "details": {
                "num_turns": len(all_logits),
                "total_tokens": len(all_log_probs),
            }
        }


# ============================================================================
# 方案 2: 从 AsyncRolloutRequest 中提取 logits（集成方案）
# ============================================================================

def extract_logits_from_rollout_request(
    request: 'AsyncRolloutRequest'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 AsyncRolloutRequest 中提取 logits 信息。
    
    这个函数展示了如何从 VERL 的 rollout 请求中获取生成的 logits。
    
    Args:
        request: AsyncRolloutRequest 对象
    
    Returns:
        (output_token_ids, log_probs) 元组
        - output_token_ids: (response_length,)
        - log_probs: (response_length,)
    
    Note:
        这个函数假设 request 已经完成了生成过程，
        并且包含了生成的 logits 信息。
    """
    if request.output_token_ids is None:
        raise ValueError("output_token_ids not available in request")
    
    if request.rollout_log_probs is None:
        # 如果没有直接的 log_probs，需要从 logits 计算
        # 这里假设有可用的 logits（需要由 rollout 引擎提供）
        raise ValueError("rollout_log_probs not available in request")
    
    return request.output_token_ids, request.rollout_log_probs


# ============================================================================
# 方案 3: 使用模型重新计算 logits（准确但计算量大）
# ============================================================================

class LogitsRecomputeHelper:
    """
    用于重新计算 logits 的辅助类。
    
    当你需要精确的 logits 但 rollout 引擎没有提供时使用。
    这种方法计算量大，但可用于 reward 的最终计算。
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        """
        Args:
            model: 语言模型，需要支持 forward pass
            device: 模型所在设备
        """
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def compute_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response_length: int,
    ) -> torch.Tensor:
        """
        重新计算生成部分的 logits。
        
        Args:
            input_ids: (seq_len,) 或 (batch_size, seq_len)
            attention_mask: 同 input_ids 形状
            position_ids: 同 input_ids 形状
            response_length: 响应部分的长度
        
        Returns:
            response_logits: (response_length, vocab_size) 或 (batch_size, response_length, vocab_size)
        """
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        logits = outputs.logits  # (seq_len, vocab_size) 或 (batch_size, seq_len, vocab_size)
        
        # 提取响应部分的 logits
        response_logits = logits[:, -response_length - 1: -1]  # 注意是 -1 的移位
        
        return response_logits
    
    @torch.no_grad()
    def compute_log_probs_from_responses(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算响应部分的 log probabilities。
        
        Args:
            input_ids: (seq_len,) 或 (batch_size, seq_len)
            attention_mask: 同 input_ids
            position_ids: 同 input_ids
            response_ids: (response_len,) 或 (batch_size, response_len) 响应 token IDs
        
        Returns:
            log_probs: (response_len,) 或 (batch_size, response_len)
        """
        response_length = response_ids.shape[-1]
        
        # 计算 logits
        response_logits = self.compute_logits(
            input_ids, attention_mask, position_ids, response_length
        )
        
        # 从 logits 计算 log probabilities
        log_probs = logprobs_from_logits(response_logits, response_ids)
        
        return log_probs
    
    @torch.no_grad()
    def compute_entropy_from_responses(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算响应部分的熵。
        
        Args:
            input_ids: (seq_len,) 或 (batch_size, seq_len)
            attention_mask: 同 input_ids
            position_ids: 同 input_ids
        
        Returns:
            entropy: (response_len,) 或 (batch_size, response_len)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        logits = outputs.logits
        entropy = entropy_from_logits(logits)
        
        return entropy


# ============================================================================
# 方案 4: 在 QAQAQA 交互的回调中使用 logits
# ============================================================================

class QAQAQAInteractionWithLogitsCallback:
    """
    展示如何在交互过程中使用 logits 的回调机制
    """
    
    def __init__(self):
        self.logits_buffer = {}
    
    def on_generation_complete(
        self,
        instance_id: str,
        turn_index: int,
        generated_answer: str,
        token_ids: torch.Tensor,
        log_probs: torch.Tensor,
        ground_truth: str,
        step_reward: float,
    ) -> Dict[str, Any]:
        """
        在每次生成完成后调用。
        
        这个回调可以在 QAQAQAInteraction.generate_response() 中调用。
        
        Args:
            instance_id: 交互实例 ID
            turn_index: 当前轮次
            generated_answer: 生成的答案文本
            token_ids: 生成的 token IDs
            log_probs: 每个 token 的 log probability
            ground_truth: 标准答案
            step_reward: 当前步的奖励
        
        Returns:
            用于传递给上层的信息字典
        """
        # 计算关键指标
        avg_log_prob = log_probs.mean().item()
        min_log_prob = log_probs.min().item()
        max_log_prob = log_probs.max().item()
        
        # 存储 logits 信息
        if instance_id not in self.logits_buffer:
            self.logits_buffer[instance_id] = {}
        
        self.logits_buffer[instance_id][turn_index] = {
            "token_ids": token_ids.cpu().numpy(),
            "log_probs": log_probs.cpu().numpy(),
            "avg_log_prob": avg_log_prob,
            "min_log_prob": min_log_prob,
            "max_log_prob": max_log_prob,
            "step_reward": step_reward,
        }
        
        return {
            "avg_log_prob": avg_log_prob,
            "min_log_prob": min_log_prob,
            "max_log_prob": max_log_prob,
            "likelihood_ratio": torch.exp(torch.tensor(avg_log_prob)).item(),  # 转换为概率
        }
    
    def finalize_with_logits(
        self,
        instance_id: str,
        step_rewards: List[float],
    ) -> Dict[str, Any]:
        """
        在交互结束时进行最终计算。
        
        使用收集的 logits 信息来计算最终奖励。
        
        Returns:
            {
                "final_reward": float,
                "logits_summary": {
                    "avg_log_prob_per_turn": [],
                    "total_tokens": int,
                    "consistency": float
                }
            }
        """
        instance_logits = self.logits_buffer.get(instance_id, {})
        
        if not instance_logits:
            return {
                "final_reward": sum(step_rewards) / len(step_rewards) if step_rewards else 0.0,
                "logits_summary": None
            }
        
        # 收集所有轮次的统计信息
        avg_log_probs = [info["avg_log_prob"] for info in instance_logits.values()]
        total_tokens = sum(len(info["log_probs"]) for info in instance_logits.values())
        
        # 计算一致性指标（所有轮次的 log prob 方差）
        consistency = 1.0 - (sum((x - sum(avg_log_probs)/len(avg_log_probs))**2 for x in avg_log_probs) / len(avg_log_probs)) ** 0.5
        
        # 基于 logits 的最终奖励加权
        weighted_reward = sum(
            reward * torch.exp(torch.tensor(log_prob)).item()
            for reward, log_prob in zip(step_rewards, avg_log_probs)
        ) / len(step_rewards)
        
        return {
            "final_reward": weighted_reward,
            "logits_summary": {
                "avg_log_prob_per_turn": avg_log_probs,
                "total_tokens": total_tokens,
                "consistency": consistency,
            }
        }


# ============================================================================
# 集成指南：如何在现有 QAQAQA 交互中使用这些方案
# ============================================================================

"""
## 集成方案

### 步骤 1: 修改 AsyncRolloutRequest 处理

在 sglang_rollout.py 的 _batch_level_generate_sequences 中：

```python
# 在生成完成后，保存 logits
if hasattr(request, 'rollout_log_probs') and request.rollout_log_probs is not None:
    await qaqaqa_interaction.store_generation_logits(
        instance_id=request.request_id,
        turn_index=current_turn,
        token_ids=request.output_token_ids,
        log_probs=request.rollout_log_probs,
    )
```

### 步骤 2: 在 generate_response 中使用 logits

在 QAQAQAInteraction.generate_response 中：

```python
# 提取最后一个 assistant 响应及其 logits
# 这里需要从某处获取 logits（通过上面的存储机制）
turn_logits = await self.get_turn_logits(instance_id, current_turn)
if turn_logits:
    # 使用 logits 调整奖励
    reward = await self._evaluate_answer_with_logits(...)
```

### 步骤 3: 计算最终奖励

在交互结束时：

```python
final_result = await interaction.compute_final_reward_with_logits(
    instance_id,
    step_rewards=all_rewards,
    logits_weights=...
)
```

## 关键文件和位置

1. **生成 logits 的位置**：
   - `verl/workers/rollout/sglang_rollout/sglang_rollout.py` 第 200-226 行
   - `_post_process_outputs()` 函数中的 `output_token_logprobs`

2. **存储 logits 的位置**：
   - `verl/workers/rollout/schemas.py` 第 115 行
   - `AsyncRolloutRequest.rollout_log_probs`

3. **计算 logits 的位置**：
   - `verl/utils/torch_functional.py` 第 65-130 行
   - `logprobs_from_logits()` 函数

4. **交互类的修改位置**：
   - `verl/interactions/multiturn_agent_interaction.py`
   - 在 `generate_response()` 方法中集成 logits 获取

## 最佳实践

1. **最小改动方案**：
   使用方案 1（QAQAQAInteractionWithLogits），在交互类中添加 logits 存储方法，
   不修改 VERL 核心代码。

2. **深度集成方案**：
   创建自定义的 rollout 类继承 SGLangRollout，
   在 `_batch_level_generate_sequences` 中增强 logits 提取。

3. **验证方案**：
   创建自定义的 Actor 类，使用 LogitsRecomputeHelper 重新计算 logits，
   用于验证和调试。

## 性能考虑

- 方案 1-2：无额外开销（logits 已由 SGLang 计算）
- 方案 3：额外的前向传播，计算量大
- 方案 4：缓存开销取决于交互数量

建议在生产环境中使用方案 1-2，在开发/调试阶段使用方案 3。
"""
