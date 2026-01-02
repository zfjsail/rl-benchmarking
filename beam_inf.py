import uuid
import math
import logging
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Dict, Optional, Tuple
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import heapq

# 配置日志
logger = logging.getLogger(__name__)

def cal_auc_map(pred,label):
    from sklearn.metrics import roc_auc_score, average_precision_score
    preds,labels = [],[]
    num_profile = 0
    overall_auc = 0
    overall_map = 0
    for aid,itm in pred.items():
        preds.append(itm)
        labels.append(label[aid])
    print("共有{}个profile".format(len(preds)))
    for p,l in zip(preds,labels):
        pos_ratio = sum(l)/len(l)
        if pos_ratio ==1 or pos_ratio<0.5:
            continue
        p = [1-r for r in p]
        l = [1-r for r in l]
        cur_auc = roc_auc_score(l,p)
        cur_map = average_precision_score(l,p)
        num_profile += 1
        overall_auc+= cur_auc
        overall_map+= cur_map
    print("共有{}个profile符合条件".format(num_profile))
    return overall_auc/num_profile , overall_map/num_profile


class BeamSearchState:
    """beam search状态类"""
    def __init__(self, task_uuid: str, cur_messages: List, score: float = 0.0):
        self.task_uuid = task_uuid
        self.cur_messages = cur_messages.copy()
        self.score = score
        self.predictions = []
        self.logits = []
    
    def __lt__(self, other):
        # 用于heapq的比较，我们需要最大堆，所以反向比较
        return self.score > other.score
    
    def copy(self):
        new_state = BeamSearchState(self.task_uuid, self.cur_messages, self.score)
        new_state.predictions = self.predictions.copy()
        new_state.logits = self.logits.copy()
        return new_state


class VLLMBatchGenerator:
    """vLLM批量生成器 with Beam Search"""
    # Yes/No token IDs
    YES_TOKEN_ID = 9454
    NO_TOKEN_ID = 2753
    
    def __init__(
        self,
        model_name: str,
        lora_path: str = None,
        tensor_parallel_size: int = 4,
        max_num_batched_tokens: int = 16384,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        beam_size: int = 4,
        turn_temperature: float = 1.0,
    ):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
            lora_path: LoRA路径
            tensor_parallel_size: 张量并行度（GPU数量）
            max_num_batched_tokens: vLLM单次推理最大tokens
            temperature: 生成文本的温度
            top_p: top-p采样
            max_tokens: 最大生成tokens
            beam_size: beam search的beam大小
            turn_temperature: 轮次间beam search的温度参数（只对Yes/No进行temperature缩放）
        """
        self.llm = LLM(
            model=model_name,
            enable_lora=True if lora_path is not None else False,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
            max_num_batched_tokens=max_num_batched_tokens,
            dtype="bfloat16",
        )
        
        if lora_path is not None:
            self.lora_request = LoRARequest(
                lora_name='lr',
                lora_path=lora_path,
                lora_int_id=1
            )
        else:
            self.lora_request = None
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=20,
        )
        
        self.beam_size = beam_size
        self.turn_temperature = turn_temperature

    def _extract_logit_from_logprobs(self, logprobs_dict):
        """
        从vLLM返回的logprobs字典中提取Yes/No token的logit
        """
        yes_logprob_obj = logprobs_dict.get(self.YES_TOKEN_ID, None)
        no_logprob_obj = logprobs_dict.get(self.NO_TOKEN_ID, None)
        yes_logprob = yes_logprob_obj.logprob if yes_logprob_obj is not None else -1e9
        no_logprob = no_logprob_obj.logprob if no_logprob_obj is not None else -1e9
        
        if yes_logprob == no_logprob == 0.0:
            logit = 0.5
        else:
            exp_yes = math.exp(yes_logprob)
            exp_no = math.exp(no_logprob)
            logit = exp_yes / (exp_yes + exp_no)

        return logit

    def _apply_turn_sampling(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        应用轮次采样策略（只使用temperature缩放）
        由于只有Yes/No两个选项，不需要top_p过滤
        
        Args:
            candidates: [(prediction, logit), ...] 列表
            
        Returns:
            过滤后的候选项列表（最多beam_size个）
        """
        if not candidates:
            return candidates
        
        # 如果候选项数量少于等于beam_size，全部保留
        if len(candidates) <= self.beam_size:
            return candidates
        
        # 应用temperature缩放logit
        scaled_logits = []
        for _, logit in candidates:
            if logit > 0 and logit < 1:
                log_odds = math.log(logit / (1 - logit))
                scaled_log_odds = log_odds / self.turn_temperature
                # 转换回概率
                scaled_logit = 1 / (1 + math.exp(-scaled_log_odds))
            else:
                scaled_logit = logit
            scaled_logits.append(scaled_logit)
        
        # 按scaled logit排序，选择top-k
        sorted_indices = sorted(
            range(len(candidates)),
            key=lambda i: scaled_logits[i],
            reverse=True
        )
        
        # 保留top beam_size个候选
        filtered = [candidates[i] for i in sorted_indices[:self.beam_size]]
        
        return filtered if filtered else [candidates[0]]

    def batch_generate_with_beamsearch(self, batch, debug_mode: bool = False, debug_task_idx: int = 0, ground_truths: List = None) -> List[List[float]]:
        """
        使用beam search进行多轮解码
        
        Args:
            batch: 输入批次
            debug_mode: 是否启用调试模式（显示所有搜索路径）
            debug_task_idx: 调试时的任务索引
        """
        # 初始化任务
        tasks = {}
        all_uuids = []
        for idx, instance in enumerate(batch):
            task_id = str(uuid.uuid4())
            all_uuids.append(task_id)
            tasks[task_id] = {}
            tasks[task_id]['messages'] = instance
            tasks[task_id]['num_turn'] = len([i for i in instance if i['role'] == 'user'])
            tasks[task_id]['user_messages'] = [i for i in instance if i['role'] == 'user']
            initial_messages = [i for i in instance if i['role'] == 'system']
            # 保存 ground truth（用于调试）
            if ground_truths is not None and idx < len(ground_truths):
                tasks[task_id]['ground_truth'] = ground_truths[idx]
            else:
                tasks[task_id]['ground_truth'] = None
            # 初始化beam search状态
            tasks[task_id]['beam_states'] = [
                BeamSearchState(task_id, initial_messages, score=0.0)
            ]
            # 用于调试的路径追踪
            tasks[task_id]['path_history'] = [[BeamSearchState(task_id, initial_messages, score=0.0)]]

        max_turn = max([tasks[task_id]['num_turn'] for task_id in tasks])
        debug_task_id = all_uuids[debug_task_idx] if debug_mode else None
        
        if debug_mode and debug_task_id:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"[DEBUG MODE] 追踪任务 {debug_task_id[:8]}... 的所有 Beam Search 路径")
            logger.debug(f"{'='*80}")
        
        # 多轮推理
        for turn in range(max_turn):
            cur_chat = {}
            turn_task_uuids = []
            
            # 为每个任务的每个beam候选准备输入
            for task_id in tasks:
                if turn < tasks[task_id]['num_turn']:
                    for state in tasks[task_id]['beam_states']:
                        # 添加当前轮的用户消息
                        state_copy = state.copy()
                        state_copy.cur_messages.append(tasks[task_id]['user_messages'][turn])
                        cur_chat[id(state_copy)] = (state_copy, state_copy.cur_messages)
                        turn_task_uuids.append((task_id, state_copy))
            
            # 批量推理
            chat_ids = [chat_id for chat_id in cur_chat.keys()]
            chats = [cur_chat[chat_id][1] for chat_id in chat_ids]
            
            results = self.llm.chat(chats, sampling_params=self.sampling_params, lora_request=self.lora_request)
            results_list = [(chat_ids[idx], results[idx]) for idx in range(len(results))]
            
            # 收集每个任务的候选项
            task_candidates = defaultdict(list)
            
            for chat_id, result_obj in results_list:
                state_copy, _ = cur_chat[chat_id]
                task_id = state_copy.task_uuid
                
                if hasattr(result_obj, 'outputs') and len(result_obj.outputs) > 0:
                    logprobs_list = result_obj.outputs[0].logprobs
                    if logprobs_list and len(logprobs_list) > 0:
                        first_token_logprobs = logprobs_list[0]
                        # 直接从 logprobs 字典中获取 Yes/No token 的 logprob
                        yes_logprob_obj = first_token_logprobs.get(self.YES_TOKEN_ID, None)
                        no_logprob_obj = first_token_logprobs.get(self.NO_TOKEN_ID, None)
                        
                        # 计算 logprobs 列表中的最小值，用作缺失 token 的 logprob
                        min_logprob = min([obj.logprob for obj in first_token_logprobs.values()]) if first_token_logprobs else -20
                        
                        yes_logprob = yes_logprob_obj.logprob if yes_logprob_obj is not None else min_logprob
                        no_logprob = no_logprob_obj.logprob if no_logprob_obj is not None else min_logprob
                    else:
                        raise ValueError(f"No logprobs found for task {task_id}")
                    
                    # 基于 logprob 直接生成两个候选：Yes 和 No
                    # 关键：直接在 logprob 空间相加分数（标准 Beam Search 方式）
                    # 同时保存 softmax 归一化后的概率用于最终评估
                    for generated_text, token_logprob in [('Yes', yes_logprob), ('No', no_logprob)]:
                        # 创建该候选的新状态副本
                        candidate_state = state_copy.copy()
                        
                        # 在 logprob 空间累加（这是 Beam Search 的标准做法）
                        new_score = candidate_state.score + token_logprob
                        
                        # 计算 softmax 归一化概率（用于最终评估）
                        exp_yes = math.exp(yes_logprob)
                        exp_no = math.exp(no_logprob)
                        normalized_prob = exp_yes / (exp_yes + exp_no)
                        
                        # 保存候选项
                        candidate_state.predictions.append(generated_text)
                        candidate_state.logits.append(normalized_prob)  # 保存归一化概率而非原始logprob
                        candidate_state.cur_messages.append({
                            'role': 'assistant',
                            'content': generated_text
                        })
                        candidate_state.score = new_score
                        
                        task_candidates[task_id].append((generated_text, token_logprob, candidate_state))
                else:
                    raise ValueError(f"No outputs found for task {task_id}")
            # 为每个任务应用beam search
            for task_id in tasks:
                if task_id in task_candidates:
                    candidates = task_candidates[task_id]
                    
                    # 提取候选的状态对象
                    candidate_states = [state for _, _, state in candidates]
                    
                    # 调试信息：排序前
                    if debug_mode and task_id == debug_task_id:
                        logger.debug(f"\n[第 {turn+1} 轮] 推理结果")
                        
                        # 打印 ground truth
                        gt = tasks[task_id].get('ground_truth')
                        if gt is not None:
                            logger.debug(f"  Ground Truth: {gt}")
                            if turn < len(gt):
                                logger.debug(f"  本轮 Ground Truth: {gt[turn]}")
                        
                        logger.debug(f"  候选项总数: {len(candidates)} (beam_size={self.beam_size} × 2 = {len(candidates)}个候选)")
                        for idx, (pred, logprob, state) in enumerate(candidates):
                            pred_history = ''.join(state.predictions)
                            logger.debug(f"    候选 {idx+1}: {pred_history:30s} | logprob={logprob:8.4f} | 累积分数={state.score:.4f}")
                        logger.debug(f"  [排序前] candidate_states 顺序和分数:")
                        for idx, state in enumerate(candidate_states):
                            pred_history = ''.join(state.predictions)
                            logger.debug(f"    {idx}: {pred_history:30s} | score={state.score:.4f}")
                    
                    # 按累积分数（logprob）排序，保留top-k（标准Beam Search）
                    # 分数越大越好，从大到小排序，选择前 beam_size 个
                    candidate_states.sort(key=lambda s: s.score, reverse=True)
                    tasks[task_id]['beam_states'] = candidate_states[:self.beam_size]
                    
                    # 调试信息：排序后
                    if debug_mode and task_id == debug_task_id:
                        logger.debug(f"  [排序后] 保留的 Top-{self.beam_size} 候选:")
                        for idx, state in enumerate(tasks[task_id]['beam_states']):
                            pred_history = ''.join(state.predictions)
                            logger.debug(f"    Beam {idx+1}: {pred_history:30s} | 累积分数={state.score:.4f}")
                    
                    # 保存路径历史
                    tasks[task_id]['path_history'].append(tasks[task_id]['beam_states'])
        
        # 提取最终结果（选择分数最高的beam）
        res_logits = []
        for task_id in all_uuids:
            best_state = tasks[task_id]['beam_states'][0]
            res_logits.append(best_state.logits)
            
            # 调试信息：显示完整的搜索树
            if debug_mode and task_id == debug_task_id:
                logger.debug(f"\n{'='*80}")
                logger.debug(f"[最终结果] 任务 {task_id[:8]}...")
                logger.debug(f"{'='*80}")
                logger.debug(f"选择的最优路径: {''.join(best_state.predictions)}")
                logger.debug(f"最终累积 logprob 分数: {best_state.score:.4f}")
                logger.debug(f"各轮 Yes 概率值: {[f'{x:.4f}' for x in best_state.logits]}")
                logger.debug(f"所有保留的 Beam 路径:")
                for beam_idx, state in enumerate(tasks[task_id]['beam_states']):
                    path_str = ''.join(state.predictions)
                    logger.debug(f"  Beam {beam_idx+1}: {path_str:30s} | 累积分数={state.score:.4f} | Yes概率={[f'{x:.4f}' for x in state.logits]}")
        
        return res_logits


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2,3 python beam_inf.py
    # ============ 主要推理脚本 ============
    
    import sys
    
    # 检查是否启用调试模式
    debug_mode = '--debug' in sys.argv
    
    test_data = pd.read_parquet('data/crossnd/test.parquet')
    batch_size = 128
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)

    llm = VLLMBatchGenerator(
        model_name='/workspace/pangyunhe/models/custom/qwen3-8b-lora',
        # lora_path='/workspace/pangyunhe/project/crossnd/verl/outputs/multiturn_grpo_v2/global_step_300/actor/lora_adapter',
        tensor_parallel_size=4,
        max_num_batched_tokens=16384,
        temperature=0,
        top_p=0.7,
        max_tokens=1,
        beam_size=4,
        turn_temperature=1.0,
    )
    
    # 配置日志级别
    if debug_mode:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
        logger.debug(f"[MODE] 调试模式已启用，将详细展示第一条数据的 Beam Search 路径")
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    for batch_idx in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data.iloc[batch_idx:batch_idx+batch_size]
        batch = batch.to_dict('records')
        
        batch_messages = []
        batch_gt = []
        batch_aids = []
        batch_pids = []
        batch_labels = []
        
        for row in batch:
            messages = row['messages']
            messages = messages.tolist()
            interaction_messages = row['extra_info']['interaction_kwargs']['messages']
            interaction_messages = interaction_messages.tolist()
            batch_messages.append(messages[:1] + interaction_messages)
            
            gt = row['reward_model']['ground_truth']
            gt = gt.tolist()
            batch_gt.append(gt)
            
            aid = row['extra_info']['metadata']['aid1']
            pids = row['extra_info']['metadata']['pids']
            pids = pids.tolist()
            batch_aids.append(aid)
            batch_pids.append(pids)
            
            batch_labels.append([1.0 if label == 'Yes' else 0.0 for label in gt])
        
        # 调试模式：只处理第一个 batch 的第一条数据
        if debug_mode:
            for i in range(10):
                batch_messages_debug = [batch_messages[i]]
                batch_gt_debug = [batch_gt[i]] if batch_gt else None
                batch_logits = llm.batch_generate_with_beamsearch(
                    batch_messages_debug, 
                    debug_mode=True, 
                    debug_task_idx=i,
                    ground_truths=batch_gt_debug
                )
            # 退出程序
            logger.debug(f"\n[完成] 调试分析结束")
            sys.exit(0)
        else:
            batch_logits = llm.batch_generate_with_beamsearch(batch_messages, debug_mode=False, ground_truths=batch_gt)
        
        for aid, pids, logits, labels in zip(batch_aids, batch_pids, batch_logits, batch_labels):
            # logits 现在是经过 softmax 归一化的概率值（0-1）
            # pids 是本条数据中各个产品对应的 pid 列表，长度等于轮次数（num_turn）
            # 因此需要按轮次索引
            for turn_idx, pid in enumerate(pids):
                if turn_idx < len(logits):
                    all_preds[aid].append(logits[turn_idx])
                else:
                    logger.error(f"错误：turn_idx({turn_idx}) 超出 logits 范围({len(logits)})")
                    all_preds[aid].append(0.5)  # 默认值
                
                if turn_idx < len(labels):
                    all_labels[aid].append(labels[turn_idx])
                else:
                    logger.error(f"错误：turn_idx({turn_idx}) 超出 labels 范围({len(labels)})")

    res = cal_auc_map(all_preds, all_labels)
    print(res)