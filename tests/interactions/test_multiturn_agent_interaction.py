import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from typing import List

# Make sure pytest-asyncio is configured
pytest_plugins = ('pytest_asyncio',)

# Import NDInteraction from verl
from verl.interactions.multiturn_agent_interaction import NDInteraction


class TestNDInteraction:
    """Test suite for NDInteraction class"""

    @pytest.fixture
    def interaction(self):
        """Create an NDInteraction instance for testing"""
        config = {}
        interaction = NDInteraction(config)
        return interaction

    @pytest.mark.asyncio
    async def test_start_interaction_basic(self, interaction):
        """Test basic interaction initialization"""
        qaqaqa_pairs = [
            {"question": "What is 2+2?", "ground_truth": "4"},
            {"question": "What is 3+3?", "ground_truth": "6"},
        ]
        instance_id = await interaction.start_interaction(
            qaqaqa_pairs=qaqaqa_pairs,
            step_reward_mode="per_step",
        )
        
        assert instance_id is not None
        assert isinstance(instance_id, str)
        
        # Verify instance state
        instance = interaction._instance_dict[instance_id]
        assert len(instance["questions"]) == 2
        assert len(instance["ground_truths"]) == 2
        assert instance["current_turn"] == 0
        assert instance["step_rewards"] == []
        assert instance["generated_answers"] == []
        assert instance["step_reward_mode"] == "per_step"

    @pytest.mark.asyncio
    async def test_start_interaction_step_reward_mode_batch(self, interaction):
        """Test interaction initialization with batch_at_end mode"""
        qaqaqa_pairs = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
        ]
        
        instance_id = await interaction.start_interaction(
            qaqaqa_pairs=qaqaqa_pairs,
            step_reward_mode="batch_at_end"
        )
        
        instance = interaction._instance_dict[instance_id]
        assert instance["step_reward_mode"] == "batch_at_end"

    @pytest.mark.asyncio
    async def test_generate_response_single_turn_per_step_mode(self, interaction):
        """Test generate_response for single turn in per_step mode"""
        qaqaqa_pairs = [
            {"question": "What is 2+2?", "ground_truth": "4"},
        ]
        
        instance_id = await interaction.start_interaction(
            qaqaqa_pairs=qaqaqa_pairs,
            step_reward_mode="per_step"
        )
        
        # Simulate correct answer (exact match)
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        
        should_terminate, response, reward, metadata = await interaction.generate_response(
            instance_id, messages
        )
        
        assert should_terminate is True
        assert reward == 1.0  # Per-step mode returns step reward immediately
        assert metadata["step_reward"] == 1.0
        assert metadata["turns_completed"] == 1
        assert metadata["total_turns"] == 1
        assert metadata["final_reward"] == 1.0  # Final reward calculated at termination

    @pytest.mark.asyncio
    async def test_generate_response_multi_turn_per_step_mode(self, interaction):
        """Test generate_response across multiple turns in per_step mode
        
        In per_step mode:
        - Each turn returns the step_reward immediately
        - final_reward is calculated but not returned during intermediate steps
        - At termination, step_reward of last step is still returned (not final_reward)
        """
        qaqaqa_pairs = [
            {"question": "Q1?", "ground_truth": "A1"},
            {"question": "Q2?", "ground_truth": "A2"},
            {"question": "Q3?", "ground_truth": "A3"},
        ]
        
        instance_id = await interaction.start_interaction(
            qaqaqa_pairs=qaqaqa_pairs,
            step_reward_mode="per_step"
        )
        
        # Turn 1 - Correct answer
        messages_1 = [
            {"role": "user", "content": "Q1?"},
            {"role": "assistant", "content": "A1"},
        ]
        should_terminate_1, response_1, reward_1, metadata_1 = await interaction.generate_response(
            instance_id, messages_1
        )
        
        assert should_terminate_1 is False
        assert reward_1 == 1.0  # Per-step mode: immediate step reward
        assert metadata_1["turns_completed"] == 1
        assert metadata_1["final_reward"] is None  # No final reward until termination
        assert "Q2" in response_1  # Next question should be in response
        
        # Turn 2 - Incorrect answer
        messages_2 = [
            {"role": "user", "content": "Q2?"},
            {"role": "assistant", "content": "Wrong"},
        ]
        should_terminate_2, response_2, reward_2, metadata_2 = await interaction.generate_response(
            instance_id, messages_2
        )
        
        assert should_terminate_2 is False
        assert reward_2 == 0.0  # Per-step mode: immediate step reward
        assert metadata_2["turns_completed"] == 2
        
        # Turn 3 - Last turn
        messages_3 = [
            {"role": "user", "content": "Q3?"},
            {"role": "assistant", "content": "A3"},
        ]
        should_terminate_3, response_3, reward_3, metadata_3 = await interaction.generate_response(
            instance_id, messages_3
        )
        
        assert should_terminate_3 is True
        assert reward_3 == 1.0  # Per-step mode: always return step reward (last answer was correct)
        assert metadata_3["turns_completed"] == 3
        assert metadata_3["all_step_rewards"] == [1.0, 0.0, 1.0]
        assert metadata_3["final_reward"] == 2.0 / 3.0  # Final reward is average, but not returned in per_step mode

    @pytest.mark.asyncio
    async def test_generate_response_multi_turn_batch_at_end_mode(self, interaction):
        """Test generate_response across multiple turns in batch_at_end mode"""
        qaqaqa_pairs = [
            {"question": "Q1?", "ground_truth": "A1"},
            {"question": "Q2?", "ground_truth": "A2"},
            {"question": "Q3?", "ground_truth": "A3"},
        ]
        
        instance_id = await interaction.start_interaction(
            qaqaqa_pairs=qaqaqa_pairs,
            step_reward_mode="batch_at_end"
        )
        
        # Turn 1 - Correct answer
        messages_1 = [
            {"role": "user", "content": "Q1?"},
            {"role": "assistant", "content": "A1"},
        ]
        should_terminate_1, response_1, reward_1, metadata_1 = await interaction.generate_response(
            instance_id, messages_1
        )
        
        assert should_terminate_1 is False
        assert reward_1 == 0.0  # Batch mode: return 0 for intermediate steps
        assert metadata_1["final_reward"] is None
        
        # Turn 2 - Incorrect answer
        messages_2 = [
            {"role": "user", "content": "Q2?"},
            {"role": "assistant", "content": "Wrong"},
        ]
        should_terminate_2, response_2, reward_2, metadata_2 = await interaction.generate_response(
            instance_id, messages_2
        )
        
        assert should_terminate_2 is False
        assert reward_2 == 0.0  # Batch mode: return 0 for intermediate steps
        
        # Turn 3 - Last turn
        messages_3 = [
            {"role": "user", "content": "Q3?"},
            {"role": "assistant", "content": "A3"},
        ]
        should_terminate_3, response_3, reward_3, metadata_3 = await interaction.generate_response(
            instance_id, messages_3
        )
        
        assert should_terminate_3 is True
        assert reward_3 == 2.0 / 3.0  # Batch mode: at termination, return average of all step rewards
        assert metadata_3["all_step_rewards"] == [1.0, 0.0, 1.0]
        assert metadata_3["final_reward"] == 2.0 / 3.0

    @pytest.mark.asyncio
    async def test_step_rewards_accumulation(self, interaction):
        """Test that step rewards are correctly accumulated"""
        qaqaqa_pairs = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
        ]
        
        instance_id = await interaction.start_interaction(qaqaqa_pairs=qaqaqa_pairs)
        
        # First answer
        messages_1 = [
            {"role": "assistant", "content": "A1"},
        ]
        await interaction.generate_response(instance_id, messages_1)
        
        # Second answer
        messages_2 = [
            {"role": "assistant", "content": "A2"},
        ]
        should_terminate, _, _, metadata = await interaction.generate_response(instance_id, messages_2)
        
        instance = interaction._instance_dict[instance_id]
        assert len(instance["step_rewards"]) == 2
        assert instance["step_rewards"][0] == 1.0
        assert instance["step_rewards"][1] == 1.0
        assert metadata["average_step_reward"] == 1.0

    @pytest.mark.asyncio
    async def test_final_reward_calculation_default(self, interaction):
        """Test final reward calculation with default function"""
        qaqaqa_pairs = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
            {"question": "Q3", "ground_truth": "A3"},
        ]
        
        instance_id = await interaction.start_interaction(qaqaqa_pairs=qaqaqa_pairs)
        
        # First answer - correct
        messages_1 = [{"role": "assistant", "content": "A1"}]
        await interaction.generate_response(instance_id, messages_1)
        
        # Second answer - incorrect
        messages_2 = [{"role": "assistant", "content": "Wrong"}]
        await interaction.generate_response(instance_id, messages_2)
        
        # Third answer - correct
        messages_3 = [{"role": "assistant", "content": "A3"}]
        should_terminate, _, current_reward, metadata = await interaction.generate_response(
            instance_id, messages_3
        )
        
        assert should_terminate is True
        # In per_step mode (default): current_reward is the step_reward of last answer (0.0)
        # final_reward is calculated but shows overall performance (2/3)
        assert current_reward == 1.0  # Step reward for incorrect answer
        assert metadata["final_reward"] == 2.0 / 3.0  # Final reward is average of all

    @pytest.mark.asyncio
    async def test_turn_history_tracking(self, interaction):
        """Test that turn history is correctly tracked"""
        qaqaqa_pairs = [
            {"question": "What is 1+1?", "ground_truth": "2"},
            {"question": "What is 2+2?", "ground_truth": "4"},
        ]
        
        instance_id = await interaction.start_interaction(qaqaqa_pairs=qaqaqa_pairs)
        
        # First turn
        messages_1 = [{"role": "assistant", "content": "2"}]
        await interaction.generate_response(instance_id, messages_1)
        
        # Second turn
        messages_2 = [{"role": "assistant", "content": "4"}]
        _, _, _, metadata = await interaction.generate_response(instance_id, messages_2)
        
        history = metadata["qa_history"]
        assert len(history) == 2
        assert history[0]["question"] == "What is 1+1?"
        assert history[0]["generated_answer"] == "2"
        assert history[0]["ground_truth"] == "2"
        assert history[0]["turn"] == 0
        assert history[0]["step_reward"] == 1.0
        assert history[1]["turn"] == 1

    @pytest.mark.asyncio
    async def test_calculate_score(self, interaction):
        """Test score calculation based on average step rewards"""
        qaqaqa_pairs = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
        ]
        
        instance_id = await interaction.start_interaction(qaqaqa_pairs=qaqaqa_pairs)
        
        # Complete interaction with mixed results
        messages_1 = [{"role": "assistant", "content": "A1"}]
        await interaction.generate_response(instance_id, messages_1)
        
        messages_2 = [{"role": "assistant", "content": "A2"}]
        await interaction.generate_response(instance_id, messages_2)
        
        score = await interaction.calculate_score(instance_id)
        assert isinstance(score, float)
        assert score == 1.0  # Both correct

    @pytest.mark.asyncio
    async def test_calculate_score_with_errors(self, interaction):
        """Test score calculation with incorrect answers"""
        qaqaqa_pairs = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
            {"question": "Q3", "ground_truth": "A3"},
        ]
        
        instance_id = await interaction.start_interaction(qaqaqa_pairs=qaqaqa_pairs)
        
        # Complete interaction with mixed results
        messages_1 = [{"role": "assistant", "content": "A1"}]
        await interaction.generate_response(instance_id, messages_1)
        
        messages_2 = [{"role": "assistant", "content": "Wrong"}]
        await interaction.generate_response(instance_id, messages_2)
        
        messages_3 = [{"role": "assistant", "content": "A3"}]
        await interaction.generate_response(instance_id, messages_3)
        
        score = await interaction.calculate_score(instance_id)
        assert score == 2.0 / 3.0  # 2 out of 3 correct

    @pytest.mark.asyncio
    async def test_finalize_interaction(self, interaction):
        """Test interaction finalization and cleanup"""
        qaqaqa_pairs = [{"question": "Q1", "ground_truth": "A1"}]
        
        instance_id = await interaction.start_interaction(qaqaqa_pairs=qaqaqa_pairs)
        assert instance_id in interaction._instance_dict
        
        await interaction.finalize_interaction(instance_id)
        assert instance_id not in interaction._instance_dict

    @pytest.mark.asyncio
    async def test_empty_qaqaqa_pairs(self, interaction):
        """Test with empty QA pairs"""
        instance_id = await interaction.start_interaction(qaqaqa_pairs=[])
        
        instance = interaction._instance_dict[instance_id]
        assert len(instance["questions"]) == 0
        assert len(instance["ground_truths"]) == 0

    @pytest.mark.asyncio
    async def test_extract_last_assistant_message(self, interaction):
        """Test extraction of last assistant message"""
        qaqaqa_pairs = [{"question": "Q1", "ground_truth": "A1"}]
        instance_id = await interaction.start_interaction(qaqaqa_pairs=qaqaqa_pairs)
        
        # Messages with multiple assistant responses
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Refine"},
            {"role": "assistant", "content": "Final response"},
        ]
        
        should_terminate, _, _, metadata = await interaction.generate_response(
            instance_id, messages
        )
        
        # Should extract the last assistant message
        assert metadata["qa_history"][0]["generated_answer"] == "Final response"

    @pytest.mark.asyncio
    async def test_default_per_step_mode(self, interaction):
        """Test that per_step is default mode when not specified"""
        qaqaqa_pairs = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
        ]
        
        instance_id = await interaction.start_interaction(qaqaqa_pairs=qaqaqa_pairs)
        
        instance = interaction._instance_dict[instance_id]
        assert instance["step_reward_mode"] == "per_step"

    @pytest.mark.asyncio
    async def test_metadata_step_reward_mode_field(self, interaction):
        """Test that metadata includes step_reward_mode information"""
        qaqaqa_pairs = [{"question": "Q1", "ground_truth": "A1"}]
        
        instance_id = await interaction.start_interaction(
            qaqaqa_pairs=qaqaqa_pairs,
            step_reward_mode="batch_at_end"
        )
        
        messages = [{"role": "assistant", "content": "A1"}]
        _, _, _, metadata = await interaction.generate_response(instance_id, messages)
        
        assert metadata["step_reward_mode"] == "batch_at_end"

    @pytest.mark.asyncio
    async def test_incorrect_answer_scoring(self, interaction):
        """Test that incorrect answers get 0.0 reward"""
        qaqaqa_pairs = [{"question": "Q1", "ground_truth": "A1"}]
        
        instance_id = await interaction.start_interaction(
            qaqaqa_pairs=qaqaqa_pairs,
            step_reward_mode="per_step"
        )
        
        messages = [{"role": "assistant", "content": "Wrong answer"}]
        should_terminate, _, reward, metadata = await interaction.generate_response(
            instance_id, messages
        )
        
        assert should_terminate is True
        assert reward == 0.0
        assert metadata["step_reward"] == 0.0
        assert metadata["final_reward"] == 0.0


if __name__ == "__main__":
    # Run tests with: pytest test_nd_interaction.py -v --asyncio-mode=auto
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])