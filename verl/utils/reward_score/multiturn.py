# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward function for multi-turn dataset.
Computes Jaccard similarity (token-level) for each turn in the conversation.
"""

from typing import Any, Optional


def jaccard_similarity(tokens1, tokens2):
    """
    Calculate Jaccard similarity between two token sequences.
    
    Jaccard similarity = |intersection| / |union|
    
    Args:
        tokens1: List of tokens from reference
        tokens2: List of tokens from generated response
        
    Returns:
        float: Jaccard similarity score in range [0, 1]
    """
    set1 = set[Any](tokens1)
    set2 = set(tokens2)
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    """
    Compute reward for multi-turn conversation based on Jaccard similarity.
    
    The solution_str should contain the generated responses, and ground_truth 
    should contain the expected responses. We compute Jaccard similarity for 
    each turn and average them.
    
    Args:
        solution_str: Generated response string
        ground_truth: Ground truth/expected response string
        extra_info: Additional information (optional)
        
    Returns:
        float: Jaccard similarity score in range [0, 1]
    """
    # Tokenize by splitting on whitespace
    solution_tokens = solution_str.split()
    ground_truth_tokens = ground_truth.split()
    
    # Compute Jaccard similarity
    similarity = jaccard_similarity(ground_truth_tokens, solution_tokens)
    
    return similarity

def strict_equal(solution_str,ground_truth,**kwargs):
    with open('./mid_data.txt','a') as f:
    # 将结果保存到本地文件中，以方便debug
        f.write(f"solution_str: {str(solution_str)}\n")
        f.write(f"ground_truth: {str(ground_truth)}\n")
        f.write(f"equal: {solution_str.lower() == ground_truth.lower()}\n\n")
    
    if solution_str.lower() == ground_truth.lower():
        return 1.0
    else:
        return 0.0