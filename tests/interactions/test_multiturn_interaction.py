#!/usr/bin/env python3
"""
Test script for the new MultiTurnInteraction class
"""

import asyncio
import sys
from verl.interactions.multiturn_interaction import MultiTurnInteraction


async def test_basic_flow():
    """Test basic flow of MultiTurnInteraction"""
    print("=" * 70)
    print("Test 1: Basic MultiTurnInteraction Flow")
    print("=" * 70)
    
    interaction = MultiTurnInteraction({})
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    
    multiturn = [
        {"role": "user", "content": "What is 3+3?"},
    ]
    
    reward_model = {
        "style": "rule",
        "ground_truth": [
            "The answer is 4.",
            "The answer is 6.",
        ]
    }
    
    # Start interaction
    instance_id = await interaction.start_interaction(
        messages=messages,
        multiturn=multiturn,
        reward_model=reward_model
    )
    print(f"✓ Started interaction with ID: {instance_id}")
    
    # Turn 1: Exact match
    turn1_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]
    
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id,
        turn1_messages
    )
    
    assert reward == 1.0, f"Expected reward 1.0, got {reward}"
    assert not should_terminate, "Should not terminate after first turn"
    assert next_prompt == "What is 3+3?", f"Unexpected next prompt: {next_prompt}"
    print(f"✓ Turn 1: Exact match - Reward: {reward}, Next prompt: '{next_prompt}'")
    
    # Turn 2: Exact match
    turn2_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "What is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]
    
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id,
        turn2_messages
    )
    
    assert reward == 1.0, f"Expected reward 1.0, got {reward}"
    assert should_terminate, "Should terminate after all turns"
    assert metadata["average_reward"] == 1.0, "Average reward should be 1.0"
    assert metadata["accuracy"] == 1.0, "Accuracy should be 1.0"
    print(f"✓ Turn 2: Exact match - Reward: {reward}, Terminated: {should_terminate}")
    print(f"  Average Reward: {metadata['average_reward']:.2f}, Accuracy: {metadata['accuracy']:.1%}")
    
    # Calculate final score
    final_score = await interaction.calculate_score(instance_id)
    assert final_score == 1.0, f"Expected final score 1.0, got {final_score}"
    print(f"✓ Final score: {final_score:.2f}")
    
    # Finalize
    await interaction.finalize_interaction(instance_id)
    print(f"✓ Interaction finalized")
    
    print()


async def test_mismatches():
    """Test with some responses not matching"""
    print("=" * 70)
    print("Test 2: MultiTurnInteraction with Mismatches")
    print("=" * 70)
    
    interaction = MultiTurnInteraction({})
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    
    multiturn = [
        {"role": "user", "content": "What is 3+3?"},
    ]
    
    reward_model = {
        "style": "rule",
        "ground_truth": [
            "The answer is 4.",
            "The answer is 6.",
        ]
    }
    
    instance_id = await interaction.start_interaction(
        messages=messages,
        multiturn=multiturn,
        reward_model=reward_model
    )
    print(f"✓ Started interaction with ID: {instance_id}")
    
    # Turn 1: Mismatch
    turn1_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 5."},  # Wrong!
    ]
    
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id,
        turn1_messages
    )
    
    assert reward == 0.0, f"Expected reward 0.0, got {reward}"
    print(f"✓ Turn 1: Mismatch - Reward: {reward}")
    
    # Turn 2: Match
    turn2_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 5."},
        {"role": "user", "content": "What is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},  # Correct!
    ]
    
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id,
        turn2_messages
    )
    
    assert reward == 1.0, f"Expected reward 1.0, got {reward}"
    assert should_terminate, "Should terminate after all turns"
    assert metadata["average_reward"] == 0.5, f"Average reward should be 0.5, got {metadata['average_reward']}"
    assert metadata["accuracy"] == 0.5, f"Accuracy should be 0.5, got {metadata['accuracy']}"
    print(f"✓ Turn 2: Match - Reward: {reward}, Terminated: {should_terminate}")
    print(f"  Average Reward: {metadata['average_reward']:.2f}, Accuracy: {metadata['accuracy']:.1%}")
    
    final_score = await interaction.calculate_score(instance_id)
    assert final_score == 0.5, f"Expected final score 0.5, got {final_score}"
    print(f"✓ Final score: {final_score:.2f}")
    
    await interaction.finalize_interaction(instance_id)
    print(f"✓ Interaction finalized")
    
    print()


async def test_whitespace_handling():
    """Test that whitespace is properly stripped"""
    print("=" * 70)
    print("Test 3: Whitespace Handling")
    print("=" * 70)
    
    interaction = MultiTurnInteraction({})
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    
    reward_model = {
        "style": "rule",
        "ground_truth": [
            "The capital of France is Paris.",
        ]
    }
    
    instance_id = await interaction.start_interaction(
        messages=messages,
        reward_model=reward_model
    )
    print(f"✓ Started interaction with ID: {instance_id}")
    
    # Response with extra whitespace
    turn_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "  The capital of France is Paris.  \n"},  # Extra spaces and newline
    ]
    
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id,
        turn_messages
    )
    
    assert reward == 1.0, f"Expected reward 1.0 (whitespace should be stripped), got {reward}"
    print(f"✓ Whitespace correctly stripped - Reward: {reward}")
    
    await interaction.finalize_interaction(instance_id)
    print(f"✓ Interaction finalized")
    
    print()


async def main():
    """Run all tests"""
    try:
        await test_basic_flow()
        await test_mismatches()
        await test_whitespace_handling()
        
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


