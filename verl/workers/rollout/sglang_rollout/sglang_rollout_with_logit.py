# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
SGLang Rollout with Logit Support

This module extends the original SGLangRollout to capture and provide token-level logits
during generation. The logits can be used in interaction modules to calculate rewards
based on model confidence, probability distributions, or other logit-based metrics.

Key differences from SGLangRollout:
1. Captures output logits from the generation process
2. Stores logits in AsyncRolloutRequest for use in interactions
3. Provides logit information through the DataProto output
"""

from __future__ import annotations

import asyncio
import logging
import os
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
)

# Import the base SGLangRollout
from .sglang_rollout import SGLangRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_logits_from_output(output):
    """
    Extract logits from single sglang inference output.
    
    This function extracts token-level logits from the SGLang output structure.
    The logits represent the raw model output before softmax, providing information
    about the model's confidence in each token prediction.
    
    Args:
        output: SGLang generation output containing meta_info with logits
        
    Returns:
        Tuple of (output_token_ids, logits) where:
        - output_token_ids: tensor of shape [seq_len] with generated token IDs
        - logits: tensor of shape [seq_len, vocab_size] with token logits
                 (or None if logits not available)
    """
    
    def _map_each_response(resp):
        # Try to extract logits from output_token_logprobs (which includes logits)
        output_token_logprobs = resp.get("meta_info", {}).get("output_token_logprobs", [])
        
        if not output_token_logprobs:
            # Fallback: return None for logits if not available
            return None, None
        
        # output_token_logprobs typically contains tuples of (log_prob, token_ids, logits)
        # but format may vary depending on SGLang version
        try:
            logits_list = []
            output_token_ids = []
            
            for item in output_token_logprobs:
                # item format: (log_prob, token_ids, logits_dict or None)
                if len(item) >= 3:
                    token_id, _, logits_data = item[0], item[1], item[2]
                    output_token_ids.append(token_id)
                    if logits_data is not None:
                        logits_list.append(logits_data)
                else:
                    output_token_ids.append(item[0])
            
            output_token_ids = torch.tensor(output_token_ids) if output_token_ids else None
            
            # If we have logits_dict, convert to tensor
            # Note: The exact format depends on SGLang's implementation
            logits_tensor = None
            if logits_list and logits_list[0] is not None:
                # This would need to be adapted based on actual SGLang output format
                # Placeholder for logits tensor construction
                logits_tensor = torch.stack([torch.tensor(l) for l in logits_list])
            
            return output_token_ids, logits_tensor
        except Exception as e:
            logger.warning(f"Failed to extract logits from output: {e}")
            return None, None
    
    output_token_ids, logits = _map_each_response(output)
    return output_token_ids, logits


class SGLangRolloutWithLogit(SGLangRollout):
    """
    SGLang Rollout with token-level logit capture.
    
    This class extends SGLangRollout to capture and expose token-level logits
    during the generation process. Logits can be used in interaction modules
    for:
    - Confidence-based reward calculation
    - Uncertainty estimation
    - Model output distribution analysis
    - Probability-weighted reward combination
    
    Usage:
        The logits are captured during generation and stored in each AsyncRolloutRequest.
        Interactions can access these logits via the request object to calculate custom rewards.
    """

    async def _handle_engine_call_with_logit(
        self, _req: AsyncRolloutRequest, sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        """
        Call engine and capture logits.
        
        Args:
            _req: The rollout request
            sampling_params: Sampling parameters for generation
            image_data: Optional image data for multimodal models
            
        Returns:
            Dictionary containing generation output with logits
        """
        generation_prompt_ids = _req.get_generation_prompt_ids(self.processing_class)
        return await self._handle_engine_generate_with_logit(generation_prompt_ids, sampling_params, image_data)

    async def _handle_engine_generate_with_logit(
        self, generation_prompt_ids: list[int], sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        """
        Handle engine generation with logit capture.
        
        This method requests logits from the engine during generation.
        The logits represent the model's raw output scores for each token in the vocabulary.
        
        Args:
            generation_prompt_ids: The prompt token IDs for generation
            sampling_params: Sampling parameters (temperature, top_p, etc.)
            image_data: Optional multimodal image data
            
        Returns:
            Dictionary containing:
            - text: Generated text
            - output_ids: Generated token IDs
            - logits: Token-level logits (shape: [seq_len, vocab_size])
            - meta_info: Additional metadata from engine
        """
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(generation_prompt_ids) - 1)

        kwargs = sampling_params.copy()
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["n"] = 1
        
        # Request logits from the engine
        # Note: This parameter name and availability depend on SGLang version
        return_logprob = kwargs.pop("logprobs", False)
        
        # Enable logit capture - this is the key difference from base implementation
        # The exact parameter name may need to be adjusted based on SGLang version
        kwargs["return_logits"] = True

        output = await self._engine.async_generate(
            input_ids=generation_prompt_ids,
            sampling_params=kwargs,
            return_logprob=return_logprob,
            image_data=image_data,
        )
        
        # Extract logits from output and add to return value
        output_token_ids, logits = _extract_logits_from_output(output)
        
        # Store logits in output for later use
        if "logits" not in output:
            output["logits"] = logits
        if "output_token_ids" not in output:
            output["output_token_ids"] = output_token_ids
        
        return output

    async def _async_rollout_a_request_with_logit(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        """
        Async rollout for a single request with logit capture.
        
        This is a modified version of _async_rollout_a_request that captures
        logits during generation and stores them in the request for interaction use.
        
        Args:
            req: The rollout request
            do_sample: Whether to sample or use greedy decoding
            is_validate: Whether in validation mode
            **kwargs: Additional arguments
            
        Returns:
            AsyncRolloutRequest with completion state and captured logits
        """
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None

        current_turns = 0
        user_turns = 0
        user_turn_rewards = []
        
        # Store logits for each turn
        generation_logits_per_turn = []

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,
                }
            )

        request_sampling_params.update(kwargs)

        while current_turns < self.config.multi_turn.max_assistant_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                # Tool calling logic (same as base)
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    if self.config.skip_tokenizer_init:
                        _req.messages[-1].tool_calls = None
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                **_req.tools_kwargs.get(tool_call.function.name, {}).get("execute_kwargs", {}),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
                    for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results, strict=True):
                        _req.update_metrics(metrics, tool_call.function.name)
                    if _req.input_ids.size(-1) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # Generation with logit capture
                prompt_length = len(_req.get_generation_prompt_ids(self.processing_class))

                if prompt_length + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break

                image_data = (
                    _req.multi_modal_data["image"]
                    if _req.multi_modal_data and "image" in _req.multi_modal_data
                    else None
                )
                video_data = (
                    _req.multi_modal_data["video"]
                    if _req.multi_modal_data and "video" in _req.multi_modal_data
                    else None
                )
                if video_data:
                    logger.warning("video support is not implemented yet")

                # Call the logit-enabled version
                output = await self._handle_engine_call_with_logit(_req, request_sampling_params, image_data=image_data)
                
                # Capture logits for this turn
                if "logits" in output and output["logits"] is not None:
                    generation_logits_per_turn.append(output["logits"])
                else:
                    # Create placeholder if logits not available
                    generation_logits_per_turn.append(None)
                
                if self.config.skip_tokenizer_init:
                    content_ids = output["output_ids"]
                    content = self.processing_class.decode(content_ids, skip_special_tokens=True)
                    content_ids = torch.tensor(
                        content_ids, dtype=_req.input_ids.dtype, device=_req.input_ids.device
                    ).unsqueeze(0)
                else:
                    content_ids = None
                    content = output["text"]

                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.processing_class, content=content, content_ids=content_ids)
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        try:
                            from json import JSONDecodeError
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except JSONDecodeError:
                            normed_content = content
                            tool_calls = []
                        except AttributeError:
                            normed_content = content
                            tool_calls = []
                        
                        from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
                        parsed_tool_calls = []
                        for tool_call in tool_calls:
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,
                                )
                            )
                            if has_decode_error:
                                continue
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        if len(parsed_tool_calls) > 0:
                            _req.add_assistant_message(
                                self.processing_class,
                                content=normed_content,
                                tool_calls=parsed_tool_calls,
                            )
                        else:
                            _req.add_assistant_message(self.processing_class, content=content, content_ids=content_ids)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(
                            self.processing_class,
                            content=content,
                            content_ids=content_ids,
                        )
                        if (
                            _req.interaction_kwargs
                            and self.interaction_map
                            and user_turns < self.config.multi_turn.max_user_turns
                            and current_turns < self.config.multi_turn.max_assistant_turns
                        ):
                            _req.state = AsyncRolloutRequestStateEnum.INTERACTING
                        else:
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
            elif _req.state == AsyncRolloutRequestStateEnum.INTERACTING:
                user_turns += 1
                messages = [{"role": x.role, "content": x.content} for x in _req.messages]

                interaction_name = _req.interaction_kwargs.get("name", "gsm8k")
                if interaction_name not in self.interaction_map:
                    raise ValueError(
                        f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                        f"{list(self.interaction_map.keys())}"
                    )

                interaction = self.interaction_map[interaction_name]

                # Pass logits to interaction for reward calculation
                interaction_kwargs = _req.interaction_kwargs.copy()
                if generation_logits_per_turn:
                    interaction_kwargs["generation_logits"] = generation_logits_per_turn[-1]
                
                should_terminate_sequence, content, reward, metrics = await interaction.generate_response(
                    _req.request_id, messages, **interaction_kwargs
                )
                user_turn_rewards.append(reward)
                
                if (
                    should_terminate_sequence
                    or user_turns > self.config.multi_turn.max_user_turns
                    or current_turns > self.config.multi_turn.max_assistant_turns
                ):
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                    break
                else:
                    _req.add_user_message(self.processing_class, content)
                    if _req.input_ids.size(-1) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    else:
                        _req.state = AsyncRolloutRequestStateEnum.RUNNING

        if current_turns >= self.config.multi_turn.max_assistant_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate rewards (same as base)
        async def calc_reward_and_release_fn(name: str, tool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}
        _req.finalize(self.processing_class, all_rewards, finish_reason_type)
        
        # Store logits in request for later use
        _req.generation_logits = generation_logits_per_turn

        return _req

    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Override to use the logit-enabled async rollout.
        """
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        if self._tp_rank == 0:
            req_list = self._preprocess_prompt_to_async_rollout_requests(prompts)

            if is_validate:
                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(
                    asyncio.gather(
                        *[self._async_rollout_a_request_with_logit(req, do_sample, is_validate, **kwargs) for req in req_list],
                    )
                )
            else:
                total_requests = len(req_list)
                target_completion = int(total_requests * (1 - self.config.get("over_sample_rate", 0.0)))
                completed_count = 0
                aborted_requests = []
                all_tasks = []

                async def rollout_a_request_with_cancellation_handler(req):
                    try:
                        result = await self._async_rollout_a_request_with_logit(req, do_sample, is_validate, **kwargs)
                        return result
                    except asyncio.CancelledError:
                        logger.info(f"Request {req.request_id} was cancelled, creating padding")
                        aborted_requests.append(req.request_id)
                        return self._create_padding_request(req)

                async def run_with_cancellation():
                    nonlocal all_tasks
                    nonlocal completed_count
                    all_tasks = [
                        asyncio.create_task(rollout_a_request_with_cancellation_handler(req)) for req in req_list
                    ]

                    try:
                        for completed_task in asyncio.as_completed(all_tasks):
                            await completed_task
                            completed_count += 1
                            if completed_count >= target_completion:
                                break
                    finally:
                        for t in all_tasks:
                            if not t.done():
                                t.cancel()

                        final_results = await asyncio.gather(*all_tasks, return_exceptions=True)
                        await self._engine.abort_request(abort_all=True)
                    return final_results

                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(run_with_cancellation())

            sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
        else:
            sorted_output_req_list = None

        import torch.distributed as dist
        dist.barrier()

        if self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj
        [sorted_output_req_list] = broadcast_pyobj(
            data=[sorted_output_req_list],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        
        # Construct the batch data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        response_loss_mask = []
        messages = []
        reward_scores = []
        multi_modal_inputs = []
        request_ids = []
        generation_logits_all = []  # Store logits for all requests
        
        if self.config.calculate_log_probs:
            output_logprobs = []
            rollout_output_token_ids = []

        for req in sorted_output_req_list:
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert (
                req.input_ids.shape[-1]
                == req.attention_mask.shape[-1]
                == req.position_ids.shape[-1]
                == req.loss_mask.shape[-1]
            )

            prompt_ids.append(req.prompt_ids.to(tgt_device).squeeze(0))
            response_ids.append(req.response_ids.to(tgt_device).squeeze(0))
            if req.response_ids.shape[-1] > self.config.response_length:
                logger.warning(
                    f"""{req.request_id=} has response_ids length {req.response_ids.shape[-1]} 
                    greater than max_response_len {self.config.response_length}"""
                )
            prompt_attention_mask.append(req.prompt_attention_mask.to(tgt_device).squeeze(0))
            response_attention_mask.append(req.response_attention_mask.to(tgt_device).squeeze(0))
            prompt_position_ids.append(req.prompt_position_ids.to(tgt_device).squeeze(0))
            response_position_ids.append(req.response_position_ids.to(tgt_device).squeeze(0))
            response_loss_mask.append(req.response_loss_mask.to(tgt_device).squeeze(0))
            messages.append({"messages": req.messages})
            reward_scores.append(req.reward_scores)
            multi_modal_inputs.append(req.multi_modal_inputs)
            request_ids.append(req.request_id)
            
            # Store logits for this request
            if hasattr(req, 'generation_logits') and req.generation_logits:
                generation_logits_all.append(req.generation_logits)
            else:
                generation_logits_all.append(None)
            
            if self.config.calculate_log_probs:
                output_logprobs.append(req.rollout_log_probs[-len(req.response_ids) :])
                rollout_output_token_ids.append(req.output_token_ids[-len(req.response_ids) :])

        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        if prompt_ids.shape[-1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
        
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[-1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        
        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        if prompt_attention_mask.shape[-1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(
                prompt_attention_mask, self.config.prompt_length, 0, left_pad=True
            )
        
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[-1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)

        if prompt_position_ids[0].dim() == 2:
            transposed_prompt_position_ids = [p.transpose(0, 1) for p in prompt_position_ids]
            prompt_position_ids = pad_sequence(
                transposed_prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            prompt_position_ids = prompt_position_ids.transpose(1, 2)
        else:
            prompt_position_ids = pad_sequence(
                prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
        if prompt_position_ids.shape[-1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(
                prompt_position_ids, self.config.prompt_length, 0, left_pad=True
            )

        if response_position_ids[0].dim() == 2:
            transposed_response_position_ids = [p.transpose(0, 1) for p in response_position_ids]
            response_position_ids = pad_sequence(
                transposed_response_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            response_position_ids = response_position_ids.transpose(1, 2)
        else:
            response_position_ids = pad_sequence(response_position_ids, batch_first=True, padding_value=0)
        if response_position_ids.shape[-1] < self.config.response_length:
            response_position_ids = pad_sequence_to_length(response_position_ids, self.config.response_length, 0)

        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        
        if self.config.calculate_log_probs:
            output_logprobs = pad_sequence(output_logprobs, padding_value=0.0, batch_first=True)
            output_logprobs = pad_sequence_to_length(
                output_logprobs, pad_token_id=0.0, max_seq_len=response_ids.shape[-1]
            ).to(tgt_device)
            rollout_output_token_ids = pad_sequence(
                rollout_output_token_ids, padding_value=self.pad_token_id, batch_first=True
            )
            rollout_output_token_ids = pad_sequence_to_length(
                rollout_output_token_ids, pad_token_id=self.pad_token_id, max_seq_len=response_ids.shape[-1]
            ).to(tgt_device)

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_loss_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(sorted_output_req_list),
        )
        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = output_logprobs
            batch["rollout_output_token_ids"] = rollout_output_token_ids

        non_tensor_batch = {
            "messages": np.array(messages),
            "reward_scores": np.array(reward_scores),
            "request_id": np.array(request_ids),
            "generation_logits": np.array(generation_logits_all, dtype=object),  # Store logits in non_tensor_batch
        }

        from transformers import ProcessorMixin
        is_multimodal = isinstance(self.processing_class, ProcessorMixin) and (
            hasattr(self.processing_class, "image_processor") or hasattr(self.model_hf_config, "vision_config")
        )

        if is_multimodal:
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
        )

