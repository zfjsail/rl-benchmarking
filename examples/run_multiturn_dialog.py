#!/usr/bin/env python
"""
Launcher script for Multi-Turn Dialog Interaction tasks.

This script demonstrates how to set up and run a complete multi-turn dialog interaction
workflow with SGLang rollout and multi-turn support.

Features:
- Loads multi-turn dialog data from examples/data_preprocess/multiturn.py
- Configures SGLang rollout with multiturn.enable=true
- Runs interaction with automatic reward calculation
- Outputs detailed statistics and logs

Usage:
    python examples/run_multiturn_dialog.py

Configuration:
    - Edit EXAMPLE_MESSAGES below to customize test data
    - Modify interaction config dict for custom behavior
    - Adjust logging level via VERL_LOGGING_LEVEL environment variable
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verl.interactions.multiturn_dialog_interaction import MultiTurnDialogInteraction

# Configure logging
logging.basicConfig(
    level=os.getenv("VERL_LOGGING_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example multi-turn dialog data
# This mimics the format from examples/data_preprocess/multiturn.py
EXAMPLE_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Write a simple Python function to calculate factorial."
    },
    {
        "role": "assistant",
        "content": (
            "```python\n"
            "def factorial(n):\n"
            "    if n == 0 or n == 1:\n"
            "        return 1\n"
            "    else:\n"
            "        return n * factorial(n-1)\n"
            "```\n\n"
            "This is a recursive function to calculate the factorial of a number."
        )
    },
    {
        "role": "user",
        "content": "Can you make it iterative instead?"
    },
    {
        "role": "assistant",
        "content": (
            "```python\n"
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(1, n+1):\n"
            "        result *= i\n"
            "    return result\n"
            "```\n\n"
            "This is an iterative version of the factorial function."
        )
    },
]


async def run_interaction_example():
    """Run a complete multi-turn dialog interaction example."""
    
    logger.info("=" * 80)
    logger.info("Multi-Turn Dialog Interaction Example")
    logger.info("=" * 80)
    
    # Step 1: Create interaction instance
    logger.info("\n[Step 1] Creating MultiTurnDialogInteraction instance...")
    interaction_config = {
        "name": "multiturn_dialog_agent"
    }
    interaction = MultiTurnDialogInteraction(interaction_config)
    logger.info("✓ Interaction instance created")
    
    # Step 2: Initialize interaction with messages
    logger.info("\n[Step 2] Initializing interaction with messages...")
    logger.info(f"Total messages: {len(EXAMPLE_MESSAGES)}")
    
    instance_id = await interaction.start_interaction(messages=EXAMPLE_MESSAGES)
    logger.info(f"✓ Interaction started with instance_id: {instance_id}")
    
    # Step 3: Simulate first turn
    logger.info("\n[Step 3] Processing Turn 1: Recursive Factorial")
    logger.info("-" * 80)
    
    # Prepare messages for Turn 1
    # In real scenario, this would come from SGLang rollout
    turn1_messages = [
        EXAMPLE_MESSAGES[0],  # system
        EXAMPLE_MESSAGES[1],  # user
        {
            "role": "assistant",
            "content": (
                "```python\n"
                "def factorial(n):\n"
                "    if n == 0 or n == 1:\n"
                "        return 1\n"
                "    else:\n"
                "        return n * factorial(n-1)\n"
                "```\n\n"
                "This is a recursive function to calculate the factorial of a number."
            )
        }
    ]
    
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id,
        turn1_messages
    )
    
    logger.info(f"Reward: {reward:.1f} (Perfect Match: {'Yes' if reward == 1.0 else 'No'})")
    logger.info(f"Should Terminate: {should_terminate}")
    logger.info(f"Average Reward: {metadata['average_reward']:.4f}")
    logger.info(f"Accuracy: {metadata['accuracy']:.1%}")
    
    # Step 4: Simulate second turn
    logger.info("\n[Step 4] Processing Turn 2: Iterative Factorial")
    logger.info("-" * 80)
    
    # Prepare messages for Turn 2
    turn2_messages = [
        EXAMPLE_MESSAGES[0],  # system
        EXAMPLE_MESSAGES[1],  # user
        EXAMPLE_MESSAGES[2],  # assistant (from turn 1)
        EXAMPLE_MESSAGES[3],  # user (new)
        {
            "role": "assistant",
            "content": (
                "```python\n"
                "def factorial(n):\n"
                "    result = 1\n"
                "    for i in range(1, n+1):\n"
                "        result *= i\n"
                "    return result\n"
                "```\n\n"
                "This is an iterative version of the factorial function."
            )
        }
    ]
    
    should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
        instance_id,
        turn2_messages
    )
    
    logger.info(f"Reward: {reward:.1f} (Perfect Match: {'Yes' if reward == 1.0 else 'No'})")
    logger.info(f"Should Terminate: {should_terminate}")
    logger.info(f"Average Reward: {metadata['average_reward']:.4f}")
    logger.info(f"Accuracy: {metadata['accuracy']:.1%}")
    
    # Step 5: Calculate final score
    logger.info("\n[Step 5] Calculating Final Score")
    logger.info("-" * 80)
    
    final_score = await interaction.calculate_score(instance_id)
    logger.info(f"Final Score: {final_score:.4f}")
    logger.info(f"Turns Completed: {metadata['turns_completed']}/{metadata['total_turns']}")
    logger.info(f"Num Matches: {metadata['num_matches']}/{metadata['total_turns']}")
    
    # Step 6: Display detailed statistics
    logger.info("\n[Step 6] Detailed Statistics")
    logger.info("-" * 80)
    
    logger.info("\nTurn History:")
    for i, turn_info in enumerate(metadata['turn_history']):
        logger.info(f"\n  Turn {i}:")
        logger.info(f"    User: {turn_info['user_message'][:60]}...")
        logger.info(f"    Match: {turn_info['match']}")
        logger.info(f"    Reward: {turn_info['reward']}")
    
    # Step 7: Finalize interaction
    logger.info("\n[Step 7] Finalizing Interaction")
    logger.info("-" * 80)
    
    await interaction.finalize_interaction(instance_id)
    logger.info("✓ Interaction finalized and resources cleaned up")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Instance ID: {instance_id}")
    logger.info(f"Total Turns: {metadata['total_turns']}")
    logger.info(f"Completed Turns: {metadata['turns_completed']}")
    logger.info(f"Perfect Matches: {metadata['num_matches']}")
    logger.info(f"Overall Accuracy: {metadata['accuracy']:.1%}")
    logger.info(f"Final Score: {final_score:.4f}")
    logger.info("=" * 80)
    
    return final_score


async def main():
    """Main entry point."""
    try:
        final_score = await run_interaction_example()
        logger.info("\n✓ Example completed successfully!")
        logger.info(f"Final Score: {final_score:.4f}")
        return 0
    except Exception as e:
        logger.error(f"\n✗ Error during execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

