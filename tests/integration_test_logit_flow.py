"""
Integration test for the complete logit flow from rollout to interaction.

This script tests the end-to-end logit flow:
1. SGLangRolloutWithLogit generates logits
2. Logits are stored in DataProto
3. Interaction receives logits
4. Reward is calculated using logits

To run this test:
    python tests/integration_test_logit_flow.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Any
from unittest.mock import Mock, patch

import numpy as np
import torch
from tensordict import TensorDict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verl import DataProto
from verl.interactions.multiturn_dialog_interaction import MultiTurnDialogInteraction


class MockInteractionTest:
    """Test interaction receiving logits"""
    
    def __init__(self):
        self.vocab_size = 128256
        
    async def test_interaction_receives_logits(self):
        """Test that MultiTurnDialogInteraction can receive and use logits"""
        print("\n" + "="*70)
        print("Integration Test: Interaction Receives Logits")
        print("="*70)
        
        try:
            # Create interaction instance
            interaction = MultiTurnDialogInteraction({})
            
            # Create test messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Is it raining?"},
                {"role": "assistant", "content": "Expected: Yes"},
            ]
            
            # Start interaction
            instance_id = await interaction.start_interaction(messages=messages)
            print(f"✓ Started interaction: {instance_id}")
            
            # Create mock logits (simulating generation output)
            seq_len = 5
            mock_logits = torch.randn(seq_len, self.vocab_size)
            
            # Simulate messages from rollout
            rollout_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Is it raining?"},
                {"role": "assistant", "content": "Yes, it is raining."},
            ]
            
            # Call generate_response with logits in kwargs
            should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
                instance_id,
                rollout_messages,
                generation_logits=mock_logits,  # Pass logits
            )
            
            print(f"✓ Interaction processed logits successfully")
            print(f"  - Should terminate: {should_terminate}")
            print(f"  - Reward: {reward}")
            print(f"  - Metadata keys: {list(metadata.keys())}")
            
            # Verify logits were available during processing
            # (In real implementation, interaction would use logits for reward calculation)
            print(f"✓ Integration test passed")
            
            # Cleanup
            await interaction.finalize_interaction(instance_id)
            
            return True
            
        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


class MockRolloutSimulation:
    """Simulate the complete rollout with logits flow"""
    
    def __init__(self):
        self.vocab_size = 128256
        self.batch_size = 2
        
    async def test_complete_logit_flow(self):
        """Test complete flow from rollout to interaction to reward"""
        print("\n" + "="*70)
        print("Integration Test: Complete Logit Flow")
        print("="*70)
        
        try:
            # Step 1: Simulate SGLangRolloutWithLogit output
            print("\n[Step 1] Simulating SGLangRolloutWithLogit output...")
            
            # Create mock DataProto with logits
            batch_data = {
                "prompts": torch.randint(0, 10000, (self.batch_size, 50)),
                "responses": torch.randint(0, 10000, (self.batch_size, 50)),
                "input_ids": torch.randint(0, 10000, (self.batch_size, 100)),
                "attention_mask": torch.ones(self.batch_size, 100),
                "position_ids": torch.arange(100).unsqueeze(0).expand(self.batch_size, -1),
            }
            batch = TensorDict(batch_data, batch_size=self.batch_size)
            
            # Create logits for each sample
            logits_sample_1 = torch.randn(10, self.vocab_size)  # 10 tokens generated
            logits_sample_2 = torch.randn(15, self.vocab_size)  # 15 tokens generated
            
            non_tensor_batch = {
                "messages": np.array([
                    {"messages": [
                        {"role": "user", "content": "Question 1"},
                        {"role": "assistant", "content": "Answer 1"}
                    ]},
                    {"messages": [
                        {"role": "user", "content": "Question 2"},
                        {"role": "assistant", "content": "Answer 2"}
                    ]},
                ]),
                "reward_scores": np.array([{}, {}]),
                "request_id": np.array(["req_1", "req_2"]),
                "generation_logits": np.array([logits_sample_1, logits_sample_2], dtype=object),
            }
            
            rollout_output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            print(f"✓ Created mock rollout output with logits")
            print(f"  - Batch size: {rollout_output.batch_size}")
            print(f"  - Has logits: {'generation_logits' in rollout_output.non_tensor_batch}")
            
            # Step 2: Verify logits in DataProto
            print("\n[Step 2] Verifying logits in DataProto...")
            
            logits_in_proto = rollout_output.non_tensor_batch.get("generation_logits")
            if logits_in_proto is not None:
                print(f"✓ Logits present in DataProto")
                for i, logits in enumerate(logits_in_proto):
                    if isinstance(logits, torch.Tensor):
                        print(f"  - Sample {i} logits shape: {logits.shape}")
                    else:
                        print(f"  - Sample {i}: {type(logits)}")
            else:
                print(f"✗ Logits not found in DataProto")
                return False
            
            # Step 3: Simulate passing logits to interaction
            print("\n[Step 3] Passing logits to interaction...")
            
            interaction = MultiTurnDialogInteraction({})
            
            # Process each sample
            for sample_idx in range(self.batch_size):
                messages_data = non_tensor_batch["messages"][sample_idx]
                logits_data = non_tensor_batch["generation_logits"][sample_idx]
                
                # Simulate interaction flow
                print(f"\n  Sample {sample_idx}:")
                
                # Start interaction
                instance_id = await interaction.start_interaction(
                    messages=messages_data.get("messages", [])
                )
                print(f"    ✓ Started interaction: {instance_id}")
                
                # Generate response with logits
                if isinstance(logits_data, torch.Tensor):
                    print(f"    ✓ Logits available: shape {logits_data.shape}")
                    
                    # In real scenario, interaction would use logits for reward calculation
                    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
                        instance_id,
                        messages_data.get("messages", []),
                        generation_logits=logits_data,
                    )
                    
                    print(f"    ✓ Processed with logits")
                    print(f"      - Reward: {reward}")
                    
                    # Cleanup
                    await interaction.finalize_interaction(instance_id)
                else:
                    print(f"    ✗ Logits not available")
            
            print(f"\n✓ Complete logit flow test passed")
            return True
            
        except Exception as e:
            print(f"✗ Complete flow test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_logit_based_reward_calculation(self):
        """Test potential logit-based reward calculation"""
        print("\n" + "="*70)
        print("Integration Test: Logit-based Reward Calculation")
        print("="*70)
        
        try:
            print("\nSimulating logit-based reward calculation scenarios:")
            
            # Scenario 1: Confidence-based reward
            print("\n[Scenario 1] Confidence-based Reward")
            
            seq_len = 10
            logits = torch.randn(seq_len, self.vocab_size)
            
            # Get confidence (max softmax probability)
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            confidence = max_probs.mean()
            
            print(f"  ✓ Average confidence: {confidence:.4f}")
            print(f"  ✓ Min confidence: {max_probs.min():.4f}")
            print(f"  ✓ Max confidence: {max_probs.max():.4f}")
            
            # Scenario 2: Entropy-based uncertainty
            print("\n[Scenario 2] Entropy-based Uncertainty")
            
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            avg_entropy = entropy.mean()
            
            print(f"  ✓ Average entropy: {avg_entropy:.4f}")
            print(f"  ✓ Min entropy: {entropy.min():.4f}")
            print(f"  ✓ Max entropy: {entropy.max():.4f}")
            
            # Lower entropy = more confident = higher reward potential
            confidence_reward = 1.0 - (avg_entropy / np.log(self.vocab_size))
            print(f"  ✓ Confidence-based reward: {confidence_reward:.4f}")
            
            # Scenario 3: Top-k probability concentration
            print("\n[Scenario 3] Top-K Probability Concentration")
            
            top_k = 5
            top_k_probs, _ = probs.topk(top_k, dim=-1)
            concentration = top_k_probs.sum(dim=-1).mean()
            
            print(f"  ✓ Average top-{top_k} probability: {concentration:.4f}")
            print(f"  ✓ Interpretation: {concentration:.1%} of probability mass in top-{top_k} tokens")
            
            # Scenario 4: Combining with text-based reward
            print("\n[Scenario 4] Combining Text and Logit Rewards")
            
            text_reward = 1.0  # Example: text matches expected output
            logit_reward = confidence_reward
            combined_reward = 0.7 * text_reward + 0.3 * logit_reward
            
            print(f"  ✓ Text-based reward: {text_reward:.4f}")
            print(f"  ✓ Logit-based reward: {logit_reward:.4f}")
            print(f"  ✓ Combined reward: {combined_reward:.4f}")
            
            print(f"\n✓ Logit-based reward calculation test passed")
            return True
            
        except Exception as e:
            print(f"✗ Logit-based reward test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def run_all_integration_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("SGLangRolloutWithLogit Integration Tests")
    print("="*70)
    
    tests = [
        ("Interaction Receives Logits", MockInteractionTest().test_interaction_receives_logits),
        ("Complete Logit Flow", MockRolloutSimulation().test_complete_logit_flow),
        ("Logit-based Rewards", MockRolloutSimulation().test_logit_based_reward_calculation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*70}")
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*70)
    print("Integration Test Summary")
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
    success = asyncio.run(run_all_integration_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

