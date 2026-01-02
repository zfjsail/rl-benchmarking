import logging
import os
import re
from typing import Any, Optional, List, Dict
from uuid import uuid4
import math
from .base import BaseInteraction
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MultiTurnInteraction(BaseInteraction):
    """Interaction for multi-turn tasks with new data format.
    
    Simplified multi-turn interaction that tracks predictions against ground truth.
    
    Data Format:
    {
        "data_source": "multiturn_1",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            ...
        ]
    }
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self._other_dict = {}
        self.YES_TOKEN_ID = 9454
        self.NO_TOKEN_ID = 2753
        self.reward_func = config.get('reward_type', 'ce')  # 'cls' or 'ce'

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Start a new multi-turn interaction instance.
        
        Args:
            instance_id: The instance ID. If None, a UUID will be generated.
            **kwargs: Additional arguments for future extensibility
        
        Returns:
            The instance ID
        """

        # print(f"debuggerinfo: interaction_kwargs: {interaction_kwargs}")
        # print(f"debuggerinfo: kwargs: {kwargs}")
        # logger.info(f"debuggerinfo: interaction_kwargs: {interaction_kwargs}")
        # logger.info(f"debuggerinfo: kwargs: {kwargs}")
        # exit() # debug


        if instance_id is None:
            instance_id = str(uuid4())
        messages = kwargs['messages']
        ground_truth = kwargs['ground_truth']   

        # Initialize: put system + first user in messages, first assistant as labels
        all_user_messages = [msg for msg in messages if msg.get("role") == "user"]
        all_assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        max_turns = len(all_assistant_messages)
        
        # Store main fields in _instance_dict (consistent with gsm8k)
        self._instance_dict[instance_id] = {
            "response": [],
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        
        # Store other fields in _other_dict
        self._other_dict[instance_id] = {
            "all_messages": messages,
            "all_user_messages": all_user_messages,
            "all_assistant_messages": all_assistant_messages,
            "current_turn": 0,
            "max_turns": max_turns,
            "turn_rewards": [],
            "turn_logprobs": [],
        }


        return instance_id
    def _calculate_reward_cls(self, res_text: str, label: str, logprobs: dict) -> tuple[float, dict]:
        """Calculate reward using classification method (cls).
        
        该方法基于完全匹配进行奖励计算，包含：
        1. 完全匹配奖励：1.0
        2. 格式奖励：0.2（如果回复是 'yes' 或 'no'）
        
        Args:
            res_text: 生成的文本（小写）
            label: 标签文本（小写）
            logprobs: 最后一个token的logprobs字典
        
        Returns:
            Tuple of (reward, metadata)
        """
        # 计算完全匹配奖励
        if label == res_text:
            turn_reward = 1.0
        else:
            turn_reward = 0.0

        # # 计算格式奖励
        # if res_text == 'no' or res_text == 'yes':
        #     format_reward = 0.2
        # else:
        #     format_reward = 0.0
        
        reward = turn_reward
        
        # 获取 YES/NO token 的 logprob 用于元数据
        yes_logprob = logprobs.get(self.YES_TOKEN_ID, None)
        no_logprob = logprobs.get(self.NO_TOKEN_ID, None)

        if yes_logprob is not None:
            yes_logprob = yes_logprob.logprob
        else:
            yes_logprob = -20.0
        if no_logprob is not None:
            no_logprob = no_logprob.logprob
        else:
            no_logprob = -20.0
        
        # 使用 softmax 计算 logit
        logit = math.exp(yes_logprob) / (math.exp(yes_logprob) + math.exp(no_logprob))
        
        metadata = {
            "logit": logit,
            "yes_logprob": yes_logprob,
            "no_logprob": no_logprob,
            "turn_reward": turn_reward,
        }
        
        return reward, metadata

    def _calculate_reward_ce(self, res_text: str, label: str, logprobs: dict) -> tuple[float, dict]:
        """Calculate reward using Cross-Entropy method (ce).
        
        该方法基于logprobs计算，包含：
        1. 提取最后一个token的所有logprob
        2. 计算YES和NO token的概率
        3. 计算二分类的logit
        4. 计算Cross Entropy Loss作为reward
        
        Args:
            res_text: 生成的文本（小写）
            label: 标签文本（小写）
            logprobs: 最后一个token的logprobs字典
        
        Returns:
            Tuple of (reward, metadata)
        """
        # 获取 YES/NO token 的 logprob
        yes_logprob = logprobs.get(self.YES_TOKEN_ID, None).logprob
        no_logprob = logprobs.get(self.NO_TOKEN_ID, None).logprob


        all_logprobs = logprobs.values()
        all_logprobs = [logprob.logprob for logprob in all_logprobs if logprob is not None]

        if label == 'yes':
            reward = np.exp(yes_logprob) / np.sum(np.exp(all_logprobs))
        elif label == 'no':
            reward = np.exp(no_logprob) / np.sum(np.exp(all_logprobs))
        else:
            raise ValueError(f"Invalid label: {label}")
        logit = yes_logprob / (yes_logprob + no_logprob)

        metadata = {
            "logit": logit,
            "yes_logprob": yes_logprob,
            "no_logprob": no_logprob,
            "reward": reward,
        }
        
        return reward, metadata

    async def generate_response(
        self,
        instance_id: str,
        cur_messages: list[dict[str, Any]],
        response_logprobs: list[float],
        **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Process generated response and return next turn.
        
        该函数包含以下功能：
        1. 解析生成的回答
        2. 根据reward_func选择计算奖励的方法（cls或ce）
        3. 在对话历史中加入 assistant 的回复
        4. 计算当前轮数和是否达到最大轮数
        5. 如果未完成，准备下一轮的对话和标签
        
        Args:
            instance_id: The instance ID
            cur_messages: 当前对话消息列表
            response_logprobs: 响应中所有token的logprobs列表
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (should_terminate, response, reward, additional_data)
        """
        instance = self._instance_dict[instance_id]
        other = self._other_dict[instance_id]
        
        # 提取最后一条消息（assistant的回复）
        response = cur_messages[-1]['content']
        res_text = response.strip().lower()
        
        # 获取当前轮的标签
        label = other["all_assistant_messages"][other["current_turn"]]['content']
        label_text = label.strip().lower()
        # 获取最后一个token的logprobs
        logprobs = response_logprobs[-1]
        # 根据reward_func选择计算方法
        if self.reward_func == 'cls':
            reward, metadata = self._calculate_reward_cls(res_text, label_text, logprobs)
        elif self.reward_func == 'ce':
            reward, metadata = self._calculate_reward_ce(res_text, label_text, logprobs)
        else:
            raise ValueError(f"Unknown reward_func: {self.reward_func}")
        
        # 保存回复
        content = ""
        for i in range(len(cur_messages) - 1, -1, -1):
            item = cur_messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
        
        instance["response"].append(content)
        
        # 保存轮次数据
        other["turn_rewards"].append(reward)
        other["turn_logprobs"].append(response_logprobs)
        
        # 移动到下一轮
        other["current_turn"] += 1
        
        # 检查是否应该终止
        should_terminate = other["current_turn"] >= other["max_turns"]
        
        # 准备下一轮的用户消息或空字符串（如果终止）
        if should_terminate:
            response = ""
        else:
            response = other["all_user_messages"][other["current_turn"]]['content']
        
        # 构造返回数据
        additional_data = {
            "turn": other["current_turn"],
            "max_turns": other["max_turns"],
            "reward": reward,
            "reward_func": self.reward_func,
            **metadata,  # 包含reward_func特定的元数据
        }
        return should_terminate, response, reward, additional_data


    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Finalize the interaction and clean up resources.
        
        Args:
            instance_id: The instance ID
            **kwargs: Additional arguments for future extensibility
        """

        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
        if instance_id in self._other_dict:
            del self._other_dict[instance_id]