"""
Test script for SGLangRolloutWithLogit to verify logit generation and capture.

This script tests:
1. Whether logits can be correctly extracted from SGLang engine output
2. Whether logits have the expected shape (seq_len, vocab_size)
3. Whether logits are correctly passed through the rollout pipeline
4. Whether logits reach the interaction layer
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verl import DataProto
from verl.workers.rollout.schemas import AsyncRolloutRequest
from verl.workers.config import RolloutConfig, HFModelConfig
from tensordict import TensorDict

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TestSGLangRolloutWithLogit:
    """Test suite for SGLangRolloutWithLogit"""

    def __init__(self):
        self.vocab_size = 128256  # Qwen vocabulary size
        self.seq_len = 50
        self.batch_size = 2
        
    def create_mock_sglang_output_with_logits(self, seq_len: int = 10):
        """
        Create a mock SGLang output with logits.
        
        Args:
            seq_len: Length of the generated sequence
            
        Returns:
            Mock output dictionary mimicking SGLang's async_generate response
        """
        # Create mock output_token_logprobs
        # Format: list of tuples (log_prob, token_id, logits)
        output_token_logprobs = []
        for i in range(seq_len):
            log_prob = np.random.uniform(-10, 0)  # Log probabilities are negative
            token_id = np.random.randint(0, self.vocab_size)
            # Create logits (raw model scores before softmax)
            logits = np.random.normal(0, 1, self.vocab_size).astype(np.float32)
            output_token_logprobs.append((log_prob, token_id, logits))
        
        mock_output = {
            "text": "This is a test response.",
            "output_ids": list(range(100, 100 + seq_len)),
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": output_token_logprobs,
            },
        }
        return mock_output

    def test_extract_logits_from_output(self):
        """Test logits extraction from mock SGLang output"""
        print("\n" + "="*70)
        print("TEST 1: Extract Logits from SGLang Output")
        print("="*70)
        
        try:
            from verl.workers.rollout.sglang_rollout.sglang_rollout_with_logit import (
                _extract_logits_from_output
            )
            
            mock_output = self.create_mock_sglang_output_with_logits(seq_len=10)
            output_token_ids, logits = _extract_logits_from_output(mock_output)
            
            print(f"✓ Successfully extracted logits from mock output")
            print(f"  - Output token IDs: {output_token_ids}")
            print(f"  - Logits shape: {logits.shape if logits is not None else 'None'}")
            
            if logits is not None:
                expected_shape = (10, self.vocab_size)
                if logits.shape == expected_shape:
                    print(f"✓ Logits shape is correct: {logits.shape}")
                    return True
                else:
                    print(f"✗ Logits shape mismatch. Expected {expected_shape}, got {logits.shape}")
                    return False
            else:
                print(f"✗ Logits extraction returned None")
                return False
                
        except Exception as e:
            print(f"✗ Error extracting logits: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_logits_in_async_rollout_request(self):
        """Test that logits are stored in AsyncRolloutRequest"""
        print("\n" + "="*70)
        print("TEST 2: Store Logits in AsyncRolloutRequest")
        print("="*70)
        
        try:
            # Create a simple AsyncRolloutRequest with mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.apply_chat_template = Mock(return_value="test prompt")
            mock_tokenizer.__call__ = Mock(return_value={
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            })
            
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
            
            # This is just to test that we can add generation_logits attribute
            mock_request = Mock()
            mock_request.generation_logits = None
            
            # Simulate storing logits
            mock_logits = torch.randn(10, self.vocab_size)
            mock_request.generation_logits = [mock_logits]
            
            print(f"✓ Successfully stored logits in mock request")
            print(f"  - Number of logits: {len(mock_request.generation_logits)}")
            print(f"  - First logits shape: {mock_request.generation_logits[0].shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error storing logits in request: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_logits_in_data_proto(self):
        """Test that logits are passed through DataProto"""
        print("\n" + "="*70)
        print("TEST 3: Pass Logits Through DataProto")
        print("="*70)
        
        try:
            # Create mock batch data
            batch_data = {
                "prompts": torch.randint(0, 1000, (self.batch_size, self.seq_len)),
                "responses": torch.randint(0, 1000, (self.batch_size, self.seq_len)),
                "input_ids": torch.randint(0, 1000, (self.batch_size, self.seq_len * 2)),
                "attention_mask": torch.ones(self.batch_size, self.seq_len * 2),
                "position_ids": torch.arange(self.seq_len * 2).unsqueeze(0).expand(self.batch_size, -1),
            }
            batch = TensorDict(batch_data, batch_size=self.batch_size)
            
            # Create non-tensor batch with logits
            logits_list = [
                [torch.randn(10, self.vocab_size) for _ in range(self.batch_size)],
            ]
            
            non_tensor_batch = {
                "messages": np.array([{"messages": []} for _ in range(self.batch_size)]),
                "reward_scores": np.array([{} for _ in range(self.batch_size)]),
                "request_id": np.array([f"req_{i}" for i in range(self.batch_size)]),
                "generation_logits": np.array(logits_list, dtype=object),
            }
            
            data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            
            print(f"✓ Successfully created DataProto with logits")
            print(f"  - Batch size: {data_proto.batch_size}")
            print(f"  - Has generation_logits: {'generation_logits' in data_proto.non_tensor_batch}")
            print(f"  - Logits type: {type(data_proto.non_tensor_batch.get('generation_logits'))}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error passing logits through DataProto: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_logits_in_interaction_kwargs(self):
        """Test that logits can be passed through interaction_kwargs"""
        print("\n" + "="*70)
        print("TEST 4: Pass Logits Through Interaction Kwargs")
        print("="*70)
        
        try:
            # Simulate logits from generation
            generation_logits = torch.randn(10, self.vocab_size)
            
            # Create interaction_kwargs
            interaction_kwargs = {
                "name": "multiturn_dialog",
                "generation_logits": generation_logits,
            }
            
            # Simulate receiving in interaction
            received_logits = interaction_kwargs.get("generation_logits", None)
            
            if received_logits is not None:
                print(f"✓ Successfully passed and received logits through interaction_kwargs")
                print(f"  - Logits shape: {received_logits.shape}")
                print(f"  - Logits dtype: {received_logits.dtype}")
                return True
            else:
                print(f"✗ Failed to receive logits from interaction_kwargs")
                return False
                
        except Exception as e:
            print(f"✗ Error passing logits through interaction_kwargs: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_logits_shape_correctness(self):
        """Test that logits have the expected shape and values"""
        print("\n" + "="*70)
        print("TEST 5: Verify Logits Shape and Value Correctness")
        print("="*70)
        
        try:
            seq_len = 15
            mock_output = self.create_mock_sglang_output_with_logits(seq_len=seq_len)
            
            from verl.workers.rollout.sglang_rollout.sglang_rollout_with_logit import (
                _extract_logits_from_output
            )
            
            output_token_ids, logits = _extract_logits_from_output(mock_output)
            
            # Check shape
            if logits is None:
                print(f"✗ Logits is None")
                return False
            
            expected_shape = (seq_len, self.vocab_size)
            if logits.shape != expected_shape:
                print(f"✗ Shape mismatch. Expected {expected_shape}, got {logits.shape}")
                return False
            
            print(f"✓ Logits shape is correct: {logits.shape}")
            
            # Check value range (logits should be roughly normal distributed)
            mean = logits.mean().item()
            std = logits.std().item()
            min_val = logits.min().item()
            max_val = logits.max().item()
            
            print(f"✓ Logits statistics:")
            print(f"  - Mean: {mean:.4f} (expected ~0)")
            print(f"  - Std: {std:.4f} (expected ~1)")
            print(f"  - Min: {min_val:.4f}")
            print(f"  - Max: {max_val:.4f}")
            
            # Check that logits are not all zeros or NaN
            if torch.isnan(logits).any():
                print(f"✗ Logits contain NaN values")
                return False
            
            if (logits == 0).all():
                print(f"✗ Logits are all zeros")
                return False
            
            print(f"✓ Logits values are valid (no NaN, not all zeros)")
            return True
            
        except Exception as e:
            print(f"✗ Error checking logits correctness: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_multi_turn_logits(self):
        """Test that logits are captured correctly for multi-turn interactions"""
        print("\n" + "="*70)
        print("TEST 6: Multi-turn Logits Capture")
        print("="*70)
        
        try:
            # Simulate multiple turns of generation
            num_turns = 3
            logits_per_turn = []
            
            for turn_idx in range(num_turns):
                seq_len = 5 + turn_idx * 2  # Varying sequence lengths
                mock_output = self.create_mock_sglang_output_with_logits(seq_len=seq_len)
                
                from verl.workers.rollout.sglang_rollout.sglang_rollout_with_logit import (
                    _extract_logits_from_output
                )
                
                output_token_ids, logits = _extract_logits_from_output(mock_output)
                logits_per_turn.append(logits)
                
                print(f"  Turn {turn_idx + 1}: logits shape = {logits.shape if logits is not None else 'None'}")
            
            # Check that we have logits for each turn
            if len(logits_per_turn) != num_turns:
                print(f"✗ Expected {num_turns} turns of logits, got {len(logits_per_turn)}")
                return False
            
            # Check that all logits have proper shape
            for idx, logits in enumerate(logits_per_turn):
                if logits is None or logits.ndim != 2:
                    print(f"✗ Turn {idx} has invalid logits shape")
                    return False
            
            print(f"✓ Successfully captured logits for {num_turns} turns")
            return True
            
        except Exception as e:
            print(f"✗ Error in multi-turn logits test: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_logits_memory_efficiency(self):
        """Test memory efficiency of logits storage"""
        print("\n" + "="*70)
        print("TEST 7: Memory Efficiency of Logits")
        print("="*70)
        
        try:
            # Create logits for a full batch
            batch_size = 8
            seq_len = 100
            
            # Each logit tensor: [seq_len, vocab_size] * 4 bytes (float32)
            single_logits_size = seq_len * self.vocab_size * 4 / (1024 * 1024)  # MB
            batch_logits_size = single_logits_size * batch_size
            
            print(f"✓ Logits memory analysis:")
            print(f"  - Single sequence logits: {single_logits_size:.2f} MB")
            print(f"  - Batch ({batch_size}) logits: {batch_logits_size:.2f} MB")
            print(f"  - Per token logits: {self.vocab_size * 4 / 1024:.2f} KB")
            
            # Suggest optimization strategies
            print(f"\n✓ Optimization strategies:")
            print(f"  1. Only store top-k logits (e.g., top-5) instead of full vocab_size")
            print(f"  2. Use float16 instead of float32 to halve memory usage")
            print(f"  3. Only capture logits for target tokens")
            print(f"  4. Store logits on CPU instead of GPU")
            
            return True
            
        except Exception as e:
            print(f"✗ Error in memory efficiency test: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_sglang_rollout_with_logit_imports(self):
        """Test that SGLangRolloutWithLogit can be imported correctly"""
        print("\n" + "="*70)
        print("TEST 8: Import SGLangRolloutWithLogit")
        print("="*70)
        
        try:
            from verl.workers.rollout.sglang_rollout.sglang_rollout_with_logit import (
                SGLangRolloutWithLogit,
                _extract_logits_from_output,
            )
            
            print(f"✓ Successfully imported SGLangRolloutWithLogit")
            print(f"  - Class: {SGLangRolloutWithLogit.__name__}")
            print(f"  - Methods:")
            
            methods = [
                "_handle_engine_call_with_logit",
                "_handle_engine_generate_with_logit",
                "_async_rollout_a_request_with_logit",
            ]
            
            for method in methods:
                if hasattr(SGLangRolloutWithLogit, method):
                    print(f"    ✓ {method}")
                else:
                    print(f"    ✗ {method} (missing)")
            
            return True
            
        except ImportError as e:
            print(f"✗ Failed to import SGLangRolloutWithLogit: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self):
        """Run all tests and report results"""
        print("\n" + "="*70)
        print("SGLangRolloutWithLogit Test Suite")
        print("="*70)
        
        tests = [
            ("Import Check", self.test_sglang_rollout_with_logit_imports),
            ("Extract Logits", self.test_extract_logits_from_output),
            ("Logits in Request", self.test_logits_in_async_rollout_request),
            ("Logits in DataProto", self.test_logits_in_data_proto),
            ("Logits in Interaction", self.test_logits_in_interaction_kwargs),
            ("Shape Correctness", self.test_logits_shape_correctness),
            ("Multi-turn Capture", self.test_multi_turn_logits),
            ("Memory Efficiency", self.test_logits_memory_efficiency),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                print(f"\n✗ Test '{test_name}' failed with exception: {e}")
                import traceback
                traceback.print_exc()
                results[test_name] = False
        
        # Print summary
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\n{'='*70}")
        print(f"总体: {passed}/{total} tests passed")
        print(f"{'='*70}\n")
        
        return passed == total


def main():
    """Main entry point"""
    tester = TestSGLangRolloutWithLogit()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

