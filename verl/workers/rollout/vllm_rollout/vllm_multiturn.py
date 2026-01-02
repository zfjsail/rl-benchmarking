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
vLLM Multi-Turn Rollout Implementation

This module implements multi-turn conversation support for vLLM, enabling:
1. Tool calling and interaction in multi-turn conversations
2. KV cache reuse across turns via Prefix Caching
3. Logits capture for reward computation
4. Compatibility with AsyncRolloutRequest (similar to SGLang)

Features:
- Automatic KV cache reuse through vLLM's Prefix Caching
- Support for tools and interactions via AsyncRolloutRequest
- Logits capture during generation for reward calculation
- Multi-turn state machine (PENDING -> RUNNING -> TOOL_CALLING/INTERACTING -> COMPLETED)
"""

import asyncio
import logging
import os
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Generator, Optional
from uuid import uuid4

import numpy as np
import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

from verl import DataProto
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
)
from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser

try:
    from sglang.srt.entrypoints.openai.protocol import Tool
except ImportError:
    from sglang.srt.openai_api.protocol import Tool

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    """Remove left padding from prompt token IDs."""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:].tolist()


def get_tool_call_parser_type(
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
) -> str:
    """Detect and return the appropriate tool call parser type for the model."""
    items = FunctionCallParser.ToolCallParserEnum.items()
    if "gpt-oss" in getattr(processing_class, "name_or_path", "").lower():
        logger.debug(f"gpt-oss model detected from name_or_path: {processing_class.name_or_path}")
        logger.debug("Using 'gpt-oss' tool call parser.")
        return "gpt-oss"
    
    for parser_type, parser_cls in items:
        parser = parser_cls()
        try:
            tokenizer_vocab = processing_class.get_vocab()
        except AttributeError:
            try:
                tokenizer_vocab = processing_class.tokenizer.get_vocab()
            except AttributeError as e:
                raise ValueError(f"Cannot get vocab from processing_class {processing_class}") from e

        if parser.bot_token.strip() in tokenizer_vocab and (
            parser.eot_token == "" or parser.eot_token.strip() in tokenizer_vocab
        ):
            return parser_type
    
    raise ValueError(f"No tool call parser found for processing_class {processing_class}")


class vLLMMultiTurnRollout(BaseRollout):
    """vLLM Multi-Turn Rollout for multi-turn conversations with tool/interaction support.
    
    This class extends BaseRollout to support multi-turn conversations similar to SGLang,
    but leverages vLLM's KV cache management and LoRA support.
    
    Key features:
    - Multi-turn state machine for conversation flow
    - Tool calling support via AsyncRolloutRequest
    - Interaction support for user message generation
    - Automatic KV cache reuse via vLLM's Prefix Caching
    - Logits capture for reward computation
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        
        self.processing_class = model_config.get_processor()
        
        # Initialize pad token ID
        try:
            self.pad_token_id = self.processing_class.pad_token_id
        except AttributeError:
            try:
                self.pad_token_id = self.processing_class.tokenizer.pad_token_id
            except AttributeError as e:
                raise ValueError(f"Cannot get pad_token_id from processing_class {self.processing_class}") from e
        
        # Initialize tools and interactions
        (
            self._tool_schemas,
            self._tool_map,
            self._tool_call_parser_type,
            self._sgl_tools,
            self._function_call_parser,
        ) = self._initialize_tools(config, self.processing_class)
        
        self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(config)
        
        logger.info(
            f"Initialized vLLMMultiTurnRollout with tools: {list(self._tool_map.keys())}, "
            f"interactions: {list(self.interaction_map.keys())}"
        )
        
        # Sampling parameters
        self._init_sampling_params(config)
        
        # Initialize vLLM engine (must be set by external initialization, typically from ray_trainer)
        self._engine = None
        self._tp_rank = 0  # Will be updated if multi-GPU is used

    def _initialize_tools(self, config, processing_class):
        """Initialize tools from configuration."""
        if config.multi_turn.tool_config_path is None:
            return [], {}, None, [], None

        tools_config_file = config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tools_config_file)

        logger.info(f"Initialize tools from configuration: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(processing_class)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(sgl_tools, tool_call_parser_type)

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    def _initialize_interactions(self, config):
        """Initialize interactions from configuration."""
        if config.multi_turn.interaction_config_path is None:
            return {}

        interaction_config_file = config.multi_turn.interaction_config_path
        interaction_map = initialize_interactions_from_config(interaction_config_file)

        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map

    def _init_sampling_params(self, config):
        """Initialize sampling parameters."""
        kwargs = {
            "n": 1,
            "max_tokens": config.response_length,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": config.get("repetition_penalty", 1.0),
        }
        
        # Support additional sampling params from config
        sampling_params_obj = SamplingParams()
        for k in config.keys():
            if k in ["temperature", "top_p", "top_k", "min_p", "best_of", "stop", "stop_token_ids"]:
                kwargs[k] = config.get(k)
            elif hasattr(sampling_params_obj, str(k)):
                kwargs[k] = config.get(k)
        
        kwargs["n"] = 1  # Already repeated in ray_trainer
        self.sampling_params = kwargs

    def set_engine(self, engine, tp_rank: int = 0):
        """Set the vLLM inference engine for this rollout.
        
        Args:
            engine: vLLM engine instance (should support async generate API).
            tp_rank: Tensor parallel rank for multi-GPU setups.
        """
        self._engine = engine
        self._tp_rank = tp_rank
        logger.info(f"vLLMMultiTurnRollout engine set with TP rank: {tp_rank}")

    @GPUMemoryLogger(role="vllm multiturn rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts in multi-turn mode.
        
        Args:
            prompts: Input batch containing prompts and metadata.
            **kwargs: Additional arguments for generation.
        
        Returns:
            DataProto: Output batch with generated sequences.
        """
        if self.config.multi_turn.enable:
            return self._req_level_generate_sequences(prompts, **kwargs)
        else:
            raise NotImplementedError("Single-turn mode not implemented for vLLMMultiTurnRollout. "
                                    "Use multi_turn.enable=True")

    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate multi-turn sequences for a batch of prompts.
        
        Each prompt is processed separately to handle tool calling and interactions.
        
        Args:
            prompts: Input batch.
            **kwargs: Additional generation arguments.
        
        Returns:
            DataProto: Output batch with multi-turn conversations.
        """
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        # Preprocess prompts to AsyncRolloutRequest objects
        req_list = self._preprocess_prompt_to_async_rollout_requests(prompts)

        # Run async rollout for each request
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
            )
        )

        sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))

        # Construct batch from sorted requests
        return self._construct_batch_from_requests(sorted_output_req_list, tgt_device)

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        """Asynchronously rollout a single request with multi-turn support.
        
        Implements the multi-turn state machine:
        PENDING -> RUNNING -> (TOOL_CALLING/INTERACTING)* -> COMPLETED
        
        Args:
            req: The rollout request.
            do_sample: Whether to use sampling.
            is_validate: Whether in validation mode.
            **kwargs: Additional generation arguments.
        
        Returns:
            Completed AsyncRolloutRequest.
        """
        assert self._engine is not None, "Engine not initialized"
        
        _req = deepcopy(req)
        finish_reason_type = None
        current_turns = 0
        user_turns = 0
        user_turn_rewards = []

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update({
                "n": 1,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
            })
        elif is_validate:
            request_sampling_params.update({
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            })

        request_sampling_params.update(kwargs)

        # Multi-turn conversation loop
        while current_turns < self.config.multi_turn.max_assistant_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                # Initialize tools and interactions
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                # Execute tool calls
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
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
                    
                    _req.add_tool_response_messages(
                        self.processing_class, 
                        [resp for resp, _, _ in tool_call_results]
                    )
                    
                    for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results, strict=True):
                        _req.update_metrics(metrics, tool_call.function.name)
                    
                    if _req.input_ids.size(-1) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling state: {_req.messages[-1]}")
            
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # Generate next response
                prompt_length = len(_req.get_generation_prompt_ids(self.processing_class))
                
                if prompt_length + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break

                # Get image data if available
                image_data = (
                    _req.multi_modal_data["image"]
                    if _req.multi_modal_data and "image" in _req.multi_modal_data
                    else None
                )

                # Generate with logits capture
                output = await self._handle_engine_generate(
                    _req, request_sampling_params, image_data=image_data
                )
                
                content = output["text"]
                content_ids = output.get("token_ids")
                finish_reason_type = FinishReasonTypeEnum.from_str(output["finish_reason"])
                current_turns += 1

                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.processing_class, content=content, content_ids=content_ids)
                    break
                else:
                    # Check for tool calls
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        
                        try:
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except (JSONDecodeError, AttributeError):
                            normed_content = content
                            tool_calls = []
                        
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
                        # No tool calls, check for interaction
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
                # Generate user response via interaction
                user_turns += 1
                messages = [{"role": x.role, "content": x.content} for x in _req.messages]

                interaction_name = _req.interaction_kwargs.get("name", "gsm8k")
                if interaction_name not in self.interaction_map:
                    raise ValueError(
                        f"Interaction '{interaction_name}' not found. Available: {list(self.interaction_map.keys())}"
                    )

                interaction = self.interaction_map[interaction_name]
                should_terminate_sequence, content, reward, metrics = await interaction.generate_response(
                    _req.request_id, messages, **_req.interaction_kwargs
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

        # Calculate tool rewards
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = [
            calc_reward_and_release_fn(name, self._tool_map[name])
            for name in _req.tools_kwargs.keys()
        ]
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}
        
        _req.finalize(self.processing_class, all_rewards, finish_reason_type)

        return _req

    async def _handle_engine_generate(
        self, 
        req: AsyncRolloutRequest, 
        sampling_params: dict, 
        image_data: Optional[list[Any]] = None
    ) -> dict:
        """Handle vLLM engine generation call with logits capture.
        
        Args:
            req: Current rollout request.
            sampling_params: Sampling parameters.
            image_data: Multi-modal image data if available.
        
        Returns:
            Dictionary with generated text, token IDs, logits, and metadata.
        """
        generation_prompt_ids = req.get_generation_prompt_ids(self.processing_class)
        
        return await self._generate_with_logits(
            generation_prompt_ids, sampling_params, image_data=image_data
        )

    async def _generate_with_logits(
        self,
        prompt_ids: list[int],
        sampling_params: dict,
        image_data: Optional[list[Any]] = None,
        request_id: str = None,
    ) -> dict:
        """Generate text with logits capture via vLLM.
        
        This method wraps vLLM's generate API to capture logits for reward computation.
        It leverages vLLM's KV cache management for multi-turn efficiency.
        
        Args:
            prompt_ids: List of prompt token IDs.
            sampling_params: Sampling parameters.
            image_data: Multi-modal image data.
            request_id: Optional request ID for tracking.
        
        Returns:
            Dictionary containing:
            - text: Generated text
            - token_ids: List of generated token IDs
            - log_probs: Log probabilities for each token
            - finish_reason: Reason for generation stop
        """
        if self._engine is None:
            raise RuntimeError("vLLM engine not initialized")

        # Prepare sampling parameters with logprobs enabled
        vllm_sampling_params = SamplingParams(**sampling_params)
        vllm_sampling_params.logprobs = 0  # Capture logprobs

        # Prepare prompt
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids,
            multi_modal_data={"image": image_data} if image_data else None
        )

        # Add LoRA request if available
        lora_request = None
        if self.model_config.lora_rank > 0:
            # Note: This requires vLLM LoRA to be pre-loaded
            # Implementation depends on your vLLM setup
            pass

        # Generate - handle both sync and async vLLM engines
        try:
            # Try async generation (vLLM v1 with async support)
            generator = self._engine.generate(
                prompt=prompt,
                sampling_params=vllm_sampling_params,
                request_id=request_id or str(uuid4()),
                lora_request=lora_request,
            )

            # Collect final output
            final_output = None
            if hasattr(generator, "__aiter__"):
                # Async generator
                async for output in generator:
                    final_output = output
            else:
                # Sync generator fallback
                for output in generator:
                    final_output = output

            if final_output is None:
                raise RuntimeError("vLLM generation failed: no output received")
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            raise

        # Extract results
        generated_token_ids = final_output.outputs[0].token_ids
        text = self.processing_class.decode(generated_token_ids, skip_special_tokens=True)

        # Extract log probabilities
        log_probs = None
        if final_output.outputs[0].logprobs:
            log_probs = []
            for i, token_id in enumerate(generated_token_ids):
                logprob_dict = final_output.outputs[0].logprobs[i]
                if token_id in logprob_dict:
                    log_probs.append(logprob_dict[token_id].logprob)
                else:
                    log_probs.append(-1.0)  # Fallback for missing logprobs

        return {
            "text": text,
            "token_ids": generated_token_ids,
            "log_probs": log_probs,
            "finish_reason": final_output.outputs[0].finish_reason or "stop",
        }

    async def _handle_pending_state(self, req: AsyncRolloutRequest) -> None:
        """Handle initialization of tools and interactions in PENDING state.
        
        Args:
            req: Current rollout request.
        """
        if req.tool_schemas is not None:
            tool_creation_coroutines = []
            for tool_schema in req.tool_schemas:
                tool = self._tool_map[tool_schema.function.name]
                create_kwargs = req.tools_kwargs[tool.name].get("create_kwargs", {})
                tool_creation_coroutines.append(tool.create(req.request_id, **create_kwargs))
            
            tool_creation_results = await asyncio.gather(*tool_creation_coroutines)
            req.add_tool_response_messages(
                self.processing_class, 
                [tool_result for _, tool_result in tool_creation_results]
            )

        if req.interaction_kwargs and self.interaction_map:
            interaction_kwargs = req.interaction_kwargs
            interaction_name = interaction_kwargs.get("name", "gsm8k")
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found. Available: {list(self.interaction_map.keys())}"
                )

            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(req.request_id, **interaction_kwargs)

    def _preprocess_prompt_to_async_rollout_requests(
        self, prompts: DataProto, n: int = 1
    ) -> list[AsyncRolloutRequest]:
        """Preprocess prompts into AsyncRolloutRequest objects.
        
        Args:
            prompts: Input batch.
            n: Number of rollouts per prompt (unused, kept for compatibility).
        
        Returns:
            List of AsyncRolloutRequest objects.
        """
        assert "raw_prompt" in prompts.non_tensor_batch, "Need data.return_raw_chat=True"
        
        req_list = []
        raw_prompts = prompts.non_tensor_batch.get("raw_prompt", [])
        multi_modal_data_list = prompts.non_tensor_batch.get(
            "multi_modal_data", [None] * len(raw_prompts)
        )
        
        batch_size = len(raw_prompts)

        for data_idx in range(batch_size):
            raw_prompt = raw_prompts[data_idx]
            multi_modal_data = multi_modal_data_list[data_idx]
            
            if self._tool_schemas:
                _tools_kwargs = prompts.non_tensor_batch.get("tools_kwargs", [{}] * batch_size)[data_idx]
                _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in _tools_kwargs.keys()]
                _input_ids = None
                _attention_mask = None
            else:
                if "input_ids" in prompts.batch and data_idx < len(prompts.batch["input_ids"]):
                    _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                    _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                else:
                    _input_ids = None
                    _attention_mask = None
                _tools_kwargs = {}
                _tool_schemas = None

            if self.interaction_map:
                _interaction_kwargs = prompts.non_tensor_batch.get("interaction_kwargs", [{}] * batch_size)[data_idx]
            else:
                _interaction_kwargs = {}

            if not isinstance(raw_prompt, (list, np.ndarray)):
                raise TypeError(f"raw_prompt must be a list or numpy array, got {type(raw_prompt)}")

            req = AsyncRolloutRequest(
                batch_data_id=data_idx,
                rollout_offset=0,
                request_id=str(uuid4()),
                state=AsyncRolloutRequestStateEnum.PENDING,
                messages=list(raw_prompt),
                multi_modal_data=multi_modal_data,
                tool_schemas=_tool_schemas,
                tools_kwargs=_tools_kwargs,
                interaction_kwargs=_interaction_kwargs,
                input_ids=_input_ids,
                response_ids=None,
                attention_mask=_attention_mask,
                response_attention_mask=None,
                response_position_ids=None,
                response_loss_mask=None,
                reward_scores={},
                max_prompt_len=self.config.prompt_length,
                max_response_len=self.config.response_length,
                max_model_len=min(
                    self.config.max_model_len,
                    self.config.prompt_length + self.config.response_length
                ),
                use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                tokenization_sanity_check_mode=self.config.multi_turn.tokenization_sanity_check_mode,
                processing_class=self.processing_class,
            )
            req_list.append(req)

        return req_list

    def _construct_batch_from_requests(
        self, sorted_output_req_list: list[AsyncRolloutRequest], tgt_device
    ) -> DataProto:
        """Construct a batch DataProto from completed requests.
        
        Args:
            sorted_output_req_list: List of completed requests.
            tgt_device: Target device for tensors.
        
        Returns:
            DataProto containing batched results.
        """
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        response_loss_mask = []
        messages = []
        reward_scores = []
        multi_modal_inputs = []
        request_ids = []

        for req in sorted_output_req_list:
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED

            prompt_ids.append(req.prompt_ids.to(tgt_device).squeeze(0))
            response_ids.append(req.response_ids.to(tgt_device).squeeze(0))
            prompt_attention_mask.append(req.prompt_attention_mask.to(tgt_device).squeeze(0))
            response_attention_mask.append(req.response_attention_mask.to(tgt_device).squeeze(0))
            prompt_position_ids.append(req.prompt_position_ids.to(tgt_device).squeeze(0))
            response_position_ids.append(req.response_position_ids.to(tgt_device).squeeze(0))
            response_loss_mask.append(req.response_loss_mask.to(tgt_device).squeeze(0))
            messages.append({"messages": req.messages})
            reward_scores.append(req.reward_scores)
            multi_modal_inputs.append(req.multi_modal_inputs)
            request_ids.append(req.request_id)

        # Pad sequences
        prompt_ids = pad_sequence(
            prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left"
        )
        if prompt_ids.shape[-1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(
                prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True
            )

        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[-1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)

        prompt_attention_mask = pad_sequence(
            prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left"
        )
        if prompt_attention_mask.shape[-1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(
                prompt_attention_mask, self.config.prompt_length, 0, left_pad=True
            )

        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[-1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(
                response_attention_mask, self.config.response_length, 0
            )

        # Handle position IDs (support multi-dimensional for models like Qwen2VL)
        if prompt_position_ids[0].dim() == 2:
            transposed = [p.transpose(0, 1) for p in prompt_position_ids]
            prompt_position_ids = pad_sequence(transposed, batch_first=True, padding_value=0, padding_side="left")
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
            transposed = [p.transpose(0, 1) for p in response_position_ids]
            response_position_ids = pad_sequence(transposed, batch_first=True, padding_value=0, padding_side="left")
            response_position_ids = response_position_ids.transpose(1, 2)
        else:
            response_position_ids = pad_sequence(response_position_ids, batch_first=True, padding_value=0)
        if response_position_ids.shape[-1] < self.config.response_length:
            response_position_ids = pad_sequence_to_length(
                response_position_ids, self.config.response_length, 0
            )

        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)

        # Construct final batch
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

        non_tensor_batch = {
            "messages": np.array(messages),
            "reward_scores": np.array(reward_scores),
            "request_id": np.array(request_ids),
        }

        is_multimodal = isinstance(self.processing_class, ProcessorMixin) and (
            hasattr(self.processing_class, "image_processor") or hasattr(self.model_hf_config, "vision_config")
        )
        if is_multimodal:
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory."""
        if not self.config.free_cache_engine:
            return
        # Implementation depends on your vLLM setup
        pass

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if not self.config.free_cache_engine:
            return
        # Implementation depends on your vLLM setup
        pass

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.
        
        Args:
            weights: Generator yielding (name, tensor) pairs.
            **kwargs: Additional arguments for weight loading.
        """
        # Implementation depends on your vLLM setup
        pass

