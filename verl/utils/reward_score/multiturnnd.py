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
Reward score computation for multiturnnd dataset.

This module aggregates the per-turn rewards calculated by the interaction layer
and returns a final score for the entire sequence.
"""


from typing import Any


import math
from sklearn.metrics import average_precision_score

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    计算多轮交互任务的最终奖励分数。
    
    参数：
    - data_source: 数据源标识
    - solution_str: 模型的最终输出
    - ground_truth: 真实标签
    - extra_info: 额外信息（包含 interaction_kwargs 等）
    - **kwargs: 其他参数（包含 tool_extra_fields，其中有 interaction_additional_data）
    
    返回：
    包含 score、turn_reward、num_turns、total_rewards 等信息的字典
    """
    # import pickle # for debug
    # with open('allinfo.pkl', 'wb') as f:
    #     pickle.dump({
    #         "data_source": data_source,
    #         "solution_str": solution_str,
    #         "ground_truth": ground_truth,
    #         "extra_info": extra_info,
    #         "kwargs": kwargs,
    #     }, f)
    YES_TOKEN_ID = 9454
    NO_TOKEN_ID = 2753

    turn_score = extra_info['turn_scores']
    all_logprobs = extra_info['all_logprobs']

    # single token logprobs
    # [{9454: Logprob(logprob=-0.0018908970523625612, rank=1, decoded_token='Yes'), 2753: Logprob(logprob=-6.376890659332275, rank=2, decoded_token='No'), 151645: Logprob(logprob=-9.376891136169434, rank=3, decoded_token='<|im_end|>'), 9693: Logprob(logprob=-10.251891136169434, rank=4, decoded_token='yes'), 785: Logprob(logprob=-11.001891136169434, rank=5, decoded_token='The')}]
    
    if all([label == 'Yes' for label in ground_truth]) or all([label == 'No' for label in ground_truth]):
        final_reward = 0.5
    else:
        labels = [1.0 if label == 'Yes' else 0.0 for label in ground_truth]
        preds = []
        for turn_idx, turn_logprobs in enumerate[Any](all_logprobs):
            yes_logprob = turn_logprobs.get(YES_TOKEN_ID, None)
            no_logprob = turn_logprobs.get(NO_TOKEN_ID, None)
            if yes_logprob is not None:
                yes_logprob = yes_logprob.logprob
            else:
                yes_logprob = 0.0
            if no_logprob is not None:
                no_logprob = no_logprob.logprob
            else:
                no_logprob = 0.0
            if yes_logprob == no_logprob == 0.0:
                pred = 0.5
            else: # softmax归一化
                exp_yes = math.exp(yes_logprob)
                exp_no = math.exp(no_logprob)
                pred = exp_yes / (exp_yes + exp_no)
            preds.append(pred)
        final_reward = average_precision_score(labels, preds)
    processed_reward = sum(turn_score) / len(turn_score)
    reward = 0.5 * processed_reward + 0.5 * final_reward
    #计算average precision score, 根据turn_logprobs和

    return {
        "score": reward,           # 最终用于训练的reward
        "turn_reward": processed_reward,  # 单轮奖励，会被记录
        "final_reward": final_reward,  # 最终奖励，会被记录
    }
