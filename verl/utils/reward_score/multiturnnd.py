import math
import numpy as np
from sklearn.metrics import average_precision_score


def compute_rank_aware_reward(labels, preds):
    """
    计算排名感知奖励,对排名靠前但预测错误的异常样本给予更大惩罚
    """
    sorted_indices = np.argsort(preds)[::-1]
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(preds)) + 1

    total_reward = 0.0
    num_anomalies = sum(labels)

    if num_anomalies == 0 or num_anomalies == len(labels):
        return 0.5

    for label, rank in zip(labels, ranks):
        if label == 1.0:
            reward_i = 1.0 / math.log(rank + 1)
        else:
            reward_i = -1.0 / math.log(rank + 1)
        total_reward += reward_i

    best_case = sum(1.0 / math.log(i + 2) for i in range(int(num_anomalies)))
    worst_case = -sum(1.0 / math.log(i + 2) for i in range(len(labels) - int(num_anomalies)))

    if best_case - worst_case == 0:
        return 0.5

    return (total_reward - worst_case) / (best_case - worst_case)


def _extract_anomaly_labels_preds(ground_truth, all_logprobs):
    """
    从 ground_truth 和 token logprobs 中提取异常标签和预测概率
    """
    YES_TOKEN_ID = 9454
    NO_TOKEN_ID = 2753

    labels = [1.0 if label == 'Yes' else 0.0 for label in ground_truth]
    preds = []

    for turn_logprobs in all_logprobs:
        yes_lp = turn_logprobs.get(YES_TOKEN_ID, None)
        no_lp = turn_logprobs.get(NO_TOKEN_ID, None)

        yes_lp = yes_lp.logprob if yes_lp is not None else 0.0
        no_lp = no_lp.logprob if no_lp is not None else 0.0

        if yes_lp == no_lp == 0.0:
            pred = 0.5
        else:
            exp_yes = math.exp(yes_lp)
            exp_no = math.exp(no_lp)
            pred = exp_yes / (exp_yes + exp_no)

        preds.append(pred)

    # Yes=正常 → No=异常
    anomaly_labels = [1.0 - l for l in labels]
    anomaly_preds = [1.0 - p for p in preds]

    return anomaly_labels, anomaly_preds


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    使用 Average Precision (AP) 作为最终奖励
    """
    turn_scores = extra_info['turn_scores']
    all_logprobs = extra_info['all_logprobs']

    if all(label == 'Yes' for label in ground_truth) or \
       all(label == 'No' for label in ground_truth):
        final_reward = 0.5
    else:
        anomaly_labels, anomaly_preds = _extract_anomaly_labels_preds(
            ground_truth, all_logprobs
        )
        final_reward = average_precision_score(anomaly_labels, anomaly_preds)

    processed_reward = sum(turn_scores) / len(turn_scores)
    reward = 0.5 * processed_reward + 0.5 * final_reward

    return {
        "score": reward,
        "turn_reward": processed_reward,
        "final_reward": final_reward,
    }


def compute_rank_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    使用 Rank-aware reward 作为最终奖励
    """
    turn_scores = extra_info['turn_scores']
    all_logprobs = extra_info['all_logprobs']

    if all(label == 'Yes' for label in ground_truth) or \
       all(label == 'No' for label in ground_truth):
        final_reward = 0.5
    else:
        anomaly_labels, anomaly_preds = _extract_anomaly_labels_preds(
            ground_truth, all_logprobs
        )
        final_reward = compute_rank_aware_reward(anomaly_labels, anomaly_preds)

    processed_reward = sum(turn_scores) / len(turn_scores)
    reward = 0.5 * processed_reward + 0.5 * final_reward

    return {
        "score": reward,
        "turn_reward": processed_reward,
        "final_reward": final_reward,
    }
