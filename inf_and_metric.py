import uuid
import math
import time
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from typing import List, Dict, Optional
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os

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



class VLLMBatchGenerator:
    """vLLM批量生成器"""
    # Yes/No token IDs
    YES_TOKEN_ID = 9454
    NO_TOKEN_ID = 2753
    def __init__(
        self,
        model_name: str,
        lora_path: str=None,
        tensor_parallel_size: int = 4,
        max_num_batched_tokens: int = 16384,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        max_logprobs: int = 1000,
        logprobs: int = 1000,
    ):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
            tensor_parallel_size: 张量并行度（GPU数量）
            max_num_batched_tokens: vLLM单次推理最大tokens
            temperature: 温度
            top_p: top-p采样
            max_tokens: 最大生成tokens
        """
        self.llm = LLM(
            model=model_name,
            enable_lora=True if lora_path is not None else False,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
            max_num_batched_tokens=max_num_batched_tokens,
            dtype="bfloat16",
            max_logprobs=max_logprobs,  # 设置最大logprobs为1000
            max_lora_rank=32
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
            logprobs=logprobs,
        )

    def _extract_logit_from_logprobs(self, logprobs_dict):
        """
        从vLLM返回的logprobs字典中提取Yes/No token的logit
        
        Args:
            logprobs_dict: vLLM返回的logprobs字典，格式为：
                          {token_id: Logprob(logprob=..., rank=..., decoded_token=...), ...}
            
        Returns:
            float: Yes和No token的softmax归一化后的logit
                   如果字典中没有该token，则对应logprob视为0（logit为0或1）
        """
        
        # 获取Yes和No token的logprobs
        # Logprob对象包含 logprob、rank、decoded_token 属性
        yes_logprob_obj = logprobs_dict.get(self.YES_TOKEN_ID,None)
        no_logprob_obj = logprobs_dict.get(self.NO_TOKEN_ID,None)

        # 提取logprob值（如果对象不存在则使用一个很小的负数，表示极低概率）
        if yes_logprob_obj is None:
            # breakpoint()
            print(f"Yes token logprob is None for token {self.YES_TOKEN_ID}")
        if no_logprob_obj is None:
            # breakpoint()
            print(f"No token logprob is None for token {self.NO_TOKEN_ID}")
        yes_logprob = yes_logprob_obj.logprob if yes_logprob_obj is not None else -20.0
        no_logprob = no_logprob_obj.logprob if no_logprob_obj is not None else -20.0
        
        # 计算softmax归一化
        # logit = exp(yes_logprob) / (exp(yes_logprob) + exp(no_logprob))
        if yes_logprob == no_logprob == 0.0:
            logit = 0.5
        else:
            exp_yes = math.exp(yes_logprob)
            exp_no = math.exp(no_logprob)
            logit = exp_yes / (exp_yes + exp_no)

        return logit

    def batch_generate(
        self,
        batch
    ) -> Dict[str, List[Dict[str, str]]]:

        tasks = {}
        all_uuids = []
        for instance in batch:
            task_uuid = str(uuid.uuid4())
            all_uuids.append(task_uuid)
            tasks[task_uuid] = {}
            tasks[task_uuid]['messages'] = instance
            tasks[task_uuid]['num_turn'] = len([i for i in instance if i['role'] == 'user'])
            tasks[task_uuid]['user_messages'] = [i for i in instance if i['role'] == 'user']
            tasks[task_uuid]['cur_messages'] = [i for i in instance if i['role'] == 'system']
            tasks[task_uuid]['pred'] = []
            tasks[task_uuid]['logit'] = []

        max_turn = max([tasks[task_uuid]['num_turn'] for task_uuid in tasks])
        
        # 多轮推理
        for turn in range(max_turn):
            cur_chat = {}
            for task_uuid in tasks:
                if turn < tasks[task_uuid]['num_turn']:
                    tasks[task_uuid]['cur_messages'].append(tasks[task_uuid]['user_messages'][turn])
                    cur_chat[task_uuid] = tasks[task_uuid]['cur_messages']
           
            chat_uuids = list(cur_chat.keys())
            chats = [cur_chat[uuid] for uuid in chat_uuids]
            # if turn == 0:
            #     results = self.llm.chat(chats, 
            #                             sampling_params=SamplingParams(
            #                                 max_tokens=1,
            #                                         temperature=self.sampling_params.temperature,
            #                                 top_p=self.sampling_params.top_p,
            #                                 logprobs=self.sampling_params.logprobs,
            #                             ),
            #                             lora_request=self.lora_request,
            #                             chat_template_kwargs={"enable_thinking": False}
            #                             )
            # else:
            results = self.llm.chat(chats, 
                                    sampling_params=self.sampling_params,
                                    lora_request=self.lora_request,
                                    chat_template_kwargs={"enable_thinking": False}
                                    )
            # breakpoint()
            # tokenizer = self.llm.get_tokenizer()
            # tokenizer.decode(results[0].prompt_token_ids)
            # 如果results是列表，按顺序对应cur_chat中的任务
            results_list = [(chat_uuids[idx], results[idx]) for idx in range(len(results))]            
            for task_uuid, result_obj in results_list:
                # 提取第一个token的logprobs（用于计算Yes/No的logit）
                logprobs_list = result_obj.outputs[0].logprobs
                if logprobs_list and len(logprobs_list) > 0:
                    # 获取第一个token的logprobs字典
                    first_token_logprobs = logprobs_list[0]
                    logit = self._extract_logit_from_logprobs(first_token_logprobs)
                else:
                    logit = 0.5  # 默认值
                if logit > 0.5:
                    generated_text='Yes'
                else:
                    generated_text='No'
                # 保存生成结果和logit
                tasks[task_uuid]['pred'].append(generated_text)
                tasks[task_uuid]['logit'].append(logit)

                # 将生成的文本加入对话历史
                tasks[task_uuid]['cur_messages'].append({
                    'role': 'assistant',
                    'content': generated_text
                })
        # res[0].outputs[0].logprobs
        res_logits = [tasks[uuid]['logit'] for uuid in all_uuids]
        return res_logits



if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=4,5,6,7 python inf_and_metric.py --lora_path  outputs/multiturn_grpo_v3/global_step_350/actor/lora_adapter
    # ============ 主要推理脚本 ============
    
    parser = argparse.ArgumentParser(description="推理和评估脚本")
    parser.add_argument("--model_name", type=str, default=None, help="模型名称/路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA适配器路径")
    parser.add_argument("--data_path", type=str, default='data/crossnd/test.parquet', help="数据路径")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="张量并行度（GPU数量）")
    parser.add_argument("--batch_size", type=int, default=128, help="批量大小")
    parser.add_argument("--save_dir", type=str, default='outputs/all_metrics.txt', help="保存目录")
    args = parser.parse_args()

    start_time = time.time()
    
    test_data = pd.read_parquet(args.data_path)
    batch_size = args.batch_size
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)

    llm = VLLMBatchGenerator(
        model_name=args.model_name,
        lora_path=args.lora_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=16384,
        temperature=0.0,
        top_p=0.7,
        max_tokens=1,
        max_logprobs=1000,
        logprobs=100,
    )

    for batch_idx in tqdm(range(0, len(test_data),batch_size)):
        batch = test_data.iloc[batch_idx:batch_idx+batch_size]
        # 转换为字典列表以提高性能和便于访问嵌套数据
        batch = batch.to_dict('records')
        
        batch_messages = []
        batch_gt = []
        batch_aids = []
        batch_pids = []
        batch_labels = []
        
        for row in batch:
            # 提取messages并转换为list（处理numpy array）
            messages = row['messages']
            messages = messages.tolist()
            if 'interaction_kwargs' in row['extra_info']:
                interaction_messages = row['extra_info']['interaction_kwargs']['messages']
                interaction_messages = interaction_messages.tolist()
                batch_messages.append(messages[:1] + interaction_messages)
                gt = row['reward_model']['ground_truth']
                gt = gt.tolist()
                batch_gt.append(gt)
            else:
                batch_messages.append(messages)
                gt = [i['content'] for i in messages if i['role'] == 'assistant']
                batch_gt.append(gt)
            
            # 提取aid和pids
            aid = row['extra_info']['metadata']['aid1']
            pids = row['extra_info']['metadata']['pids']

            pids = pids.tolist()
            batch_aids.append(aid)
            batch_pids.append(pids)
            
            # 处理标签
            batch_labels.append([1.0 if label == 'Yes' else 0.0 for label in gt])
        
        batch_logits = llm.batch_generate(batch_messages)
        for aid, pids, logits, labels in zip(batch_aids, batch_pids, batch_logits, batch_labels):
            for idxs,pid in enumerate(pids):
                all_preds[aid].append(logits[idxs])
                all_labels[aid].append(labels[idxs])

    # with open('outputs/pred/pred.json','w') as f:
    #     import json
    #     json.dump({"preds":all_preds,"labels":all_labels},f)


    res = cal_auc_map(all_preds,all_labels)
    print(res)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"推理总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    
    # 保存评估结果到文件（追加方式）
    # os.makedirs(args.save_dir, exist_ok=True)

    with open(args.save_dir, "a") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"LoRA Path: {args.lora_path}\n")
        f.write(f"Tensor Parallel Size: {args.tensor_parallel_size}\n")
        f.write(f"AUC: {res[0]:.6f}, MAP: {res[1]:.6f}\n")
        f.write(f"推理总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)\n")
        f.write("-" * 80 + "\n")

    # llm = LLM(
    #     model='/workspace/pangyunhe/models/custom/qwen3-8b-lora',
    #     tensor_parallel_size=1,
    #     enable_prefix_caching=True,
    #     max_num_batched_tokens=16384,
    #     dtype="bfloat16",
    # )


    # sampling_params = SamplingParams(
    #     temperature=0.0,
    #     top_p=1.0,
    #     max_tokens=1,
    #     logprobs=5,
    # )
    # tmp_message =  [{'role': 'system', 'content': 'You are helpful'},{'role': 'user', 'content': 'Hello'},{'role': 'assistant', 'content': 'Hello, how can I help you today?'},{'role': 'user', 'content': 'How are you?'}]
    # res = llm.chat(tmp_message,sampling_params=sampling_params )
    # breakpoint()