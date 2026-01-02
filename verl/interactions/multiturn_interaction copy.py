import logging
import os
import re
from typing import Any, Optional, List, Dict
from uuid import uuid4
import math
from .base import BaseInteraction

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
        self.reward_func = 'ce' # or cls

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
    async def generate_response(
        self,
        instance_id: str,
        cur_messages: list[dict[str, Any]],
        response_logprobs: list[float],
        **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Process generated response and return next turn.
        
        该函数包含以下功能：
        1. 解析生成的回答，标准化为 'Yes', 'No', 或 'Unknown'
        2. 与标签进行对比，计算奖励（正确为1.0，错误为0.0）
        3. 在对话历史中加入 assistant 的回复
        4. 计算当前轮数和是否达到最大轮数
        5. 如果未完成，准备下一轮的对话和标签
        
        Args:
            instance_id: The instance ID
            messages: Full conversation messages
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (should_terminate, response, reward, additional_data)
        """
        # if self._other_dict[instance_id]['current_turn'] >=5:
        #     with open('interaction.pkl','wb') as f:
        #         import pickle
        #         pickle.dump({
        #             "cur_messages":cur_messages,
        #             "response_logprobs":response_logprobs,
        #             "kwargs":kwargs
        #         }, f)
        #     exit(0)
        instance = self._instance_dict[instance_id]
        other = self._other_dict[instance_id]
        
        response = cur_messages[-1]['content']


        res_text = response.strip().lower()
        label = other["all_assistant_messages"][other["current_turn"]]['content'].strip().lower()

        if label == res_text:
            turn_reward = 1
        else:
            turn_reward = 0

        if res_text == 'no' or res_text =='yes':
            format_reward = 0.2
        else:
            format_reward = 0
        reward = turn_reward + format_reward



        logprobs = response_logprobs[-1]
        yes_logprob = logprobs.get(self.YES_TOKEN_ID, None)
        no_logprob = logprobs.get(self.NO_TOKEN_ID, None)

        if yes_logprob is not None:
            yes_logprob = yes_logprob.logprob
        else:
            yes_logprob = -20.0   # previously 0.0  
        if no_logprob is not None:
            no_logprob = no_logprob.logprob
        else:
            no_logprob = -20.0   # previously 0.0 


        content = ""
        for i in range(len(cur_messages) - 1, -1, -1):
            item = cur_messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
        
        instance["response"].append(content)
                
        
        label = other["all_assistant_messages"][other["current_turn"]]['content']
        if label == 'Yes':
            label = 1.0
        elif label == 'No':
            label = 0.0
        else:
            raise ValueError(f"Invalid label: {label}")

        logit = yes_logprob / (yes_logprob + no_logprob) 
        # reward = logit * label + (1 - logit) * (1 - label) 


        # 第三步：计算奖励
        # reward = self._calculate_reward(parsed_response, label)

        other["turn_rewards"].append(reward)
        other["turn_logprobs"].append(response_logprobs)
        # 第五步：移动到下一轮
        other["current_turn"] += 1
        
        # 第六步：检查是否应该终止
        should_terminate = other["current_turn"] >= other["max_turns"]
        
        # 准备返回的响应和附加数据
        if should_terminate:
            response = ""
        else:
            response = other["all_user_messages"][other["current_turn"]]['content']
        additional_data = {
            "turn": other["current_turn"],
            "max_turns": other["max_turns"],
            "logit": logit,
            "label": label,
            "reward": reward,
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