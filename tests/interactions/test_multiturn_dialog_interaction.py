#!/usr/bin/env python
"""
Unit tests for MultiTurnDialogInteraction.

Tests cover:
- Interaction initialization
- Message parsing
- Response generation and evaluation
- Score calculation
- Error handling
"""

import asyncio
import pytest
from uuid import uuid4

from verl.interactions.multiturn_dialog_interaction import MultiTurnDialogInteraction


class TestMultiTurnDialogInteraction:
    """Test suite for MultiTurnDialogInteraction."""
    
    @pytest.fixture
    def interaction(self):
        """Create interaction instance for testing."""
        return MultiTurnDialogInteraction({})
    
    @pytest.fixture
    def sample_messages(self):
        """Sample multi-turn dialog messages."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "3+3 equals 6."},
        ]
    
    # ==================== Test start_interaction ====================
    
    @pytest.mark.asyncio
    async def test_start_interaction_with_valid_messages(self, interaction, sample_messages):
        """Test starting interaction with valid messages."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        assert instance_id is not None
        assert instance_id in interaction._instance_dict
        
        instance = interaction._instance_dict[instance_id]
        assert len(instance["dialog_turns"]) == 2  # Two Q&A pairs
        assert instance["current_turn"] == 0
    
    @pytest.mark.asyncio
    async def test_start_interaction_with_custom_id(self, interaction, sample_messages):
        """Test starting interaction with custom instance ID."""
        custom_id = "custom_instance_123"
        instance_id = await interaction.start_interaction(
            instance_id=custom_id,
            messages=sample_messages
        )
        assert instance_id == custom_id
    
    @pytest.mark.asyncio
    async def test_start_interaction_with_no_messages(self, interaction):
        """Test starting interaction with no messages."""
        instance_id = await interaction.start_interaction(messages=None)
        assert instance_id is not None
        
        instance = interaction._instance_dict[instance_id]
        assert len(instance["dialog_turns"]) == 0
    
    @pytest.mark.asyncio
    async def test_start_interaction_with_unpaired_user_message(self, interaction):
        """Test handling unpaired user messages."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},  # No answer
        ]
        
        instance_id = await interaction.start_interaction(messages=messages)
        instance = interaction._instance_dict[instance_id]
        
        # Only one complete pair should be extracted
        assert len(instance["dialog_turns"]) == 1
    
    @pytest.mark.asyncio
    async def test_start_interaction_with_system_message(self, interaction):
        """Test system message extraction."""
        messages = [
            {"role": "system", "content": "Custom system prompt"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        
        instance_id = await interaction.start_interaction(messages=messages)
        instance = interaction._instance_dict[instance_id]
        
        assert instance["system_message"] == "Custom system prompt"
    
    # ==================== Test generate_response ====================
    
    @pytest.mark.asyncio
    async def test_generate_response_exact_match(self, interaction, sample_messages):
        """Test response evaluation with exact match."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        
        # Generate response that exactly matches
        generated_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]
        
        should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
            instance_id,
            generated_messages
        )
        
        assert reward == 1.0
        assert should_terminate is False
        assert metadata["current_turn_reward"] == 1.0
        assert metadata["accuracy"] == 1.0
    
    @pytest.mark.asyncio
    async def test_generate_response_no_match(self, interaction, sample_messages):
        """Test response evaluation with no match."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        
        # Generate response that doesn't match
        generated_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 5."},  # Wrong answer
        ]
        
        should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
            instance_id,
            generated_messages
        )
        
        assert reward == 0.0
        assert should_terminate is False
        assert metadata["current_turn_reward"] == 0.0
        assert metadata["accuracy"] == 0.0
    
    @pytest.mark.asyncio
    async def test_generate_response_multiple_turns(self, interaction, sample_messages):
        """Test multiple consecutive responses."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        
        # Turn 1: Exact match
        generated_messages_1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]
        should_terminate_1, _, reward_1, metadata_1 = await interaction.generate_response(
            instance_id, generated_messages_1
        )
        
        assert reward_1 == 1.0
        assert should_terminate_1 is False
        assert metadata_1["turns_completed"] == 1
        assert metadata_1["average_reward"] == 1.0
        
        # Turn 2: Exact match
        generated_messages_2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "3+3 equals 6."},
        ]
        should_terminate_2, _, reward_2, metadata_2 = await interaction.generate_response(
            instance_id, generated_messages_2
        )
        
        assert reward_2 == 1.0
        assert should_terminate_2 is True  # All turns completed
        assert metadata_2["turns_completed"] == 2
        assert metadata_2["accuracy"] == 1.0
        assert metadata_2["num_matches"] == 2
    
    @pytest.mark.asyncio
    async def test_generate_response_mixed_accuracy(self, interaction, sample_messages):
        """Test mixed accuracy across turns."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        
        # Turn 1: No match
        generated_messages_1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals wrong."},
        ]
        _, _, reward_1, _ = await interaction.generate_response(instance_id, generated_messages_1)
        assert reward_1 == 0.0
        
        # Turn 2: Match
        generated_messages_2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals wrong."},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "3+3 equals 6."},
        ]
        _, _, reward_2, metadata = await interaction.generate_response(instance_id, generated_messages_2)
        
        assert reward_2 == 1.0
        assert metadata["average_reward"] == 0.5
        assert metadata["accuracy"] == 0.5
        assert metadata["num_matches"] == 1
    
    @pytest.mark.asyncio
    async def test_generate_response_yes_no_matching(self, interaction):
        """Test Yes/No answer matching across multiple turns."""
        # Create messages with Yes/No expected responses
        messages = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Is the sky blue?"},
            {"role": "assistant", "content": "Yes"},
            {"role": "user", "content": "Is the grass red?"},
            {"role": "assistant", "content": "No"},
        ]
        
        instance_id = await interaction.start_interaction(messages=messages)
        
        # Turn 1: Generated response contains "Yes", expected is "Yes" -> reward 1.0
        generated_messages_1 = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Is the sky blue?"},
            {"role": "assistant", "content": "Yes, the sky is blue."},
        ]
        should_terminate_1, _, reward_1, metadata_1 = await interaction.generate_response(
            instance_id, generated_messages_1
        )
        
        assert reward_1 == 1.0, f"Turn 1: Expected reward 1.0 for matching Yes, got {reward_1}"
        assert should_terminate_1 is False
        assert metadata_1["current_turn_reward"] == 1.0
        assert metadata_1["turns_completed"] == 1
        assert metadata_1["average_reward"] == 1.0
        
        # Turn 2: Generated response contains "No", expected is "No" -> reward 1.0
        generated_messages_2 = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Is the sky blue?"},
            {"role": "assistant", "content": "Yes, the sky is blue."},
            {"role": "user", "content": "Is the grass red?"},
            {"role": "assistant", "content": "No, the grass is green."},
        ]
        should_terminate_2, _, reward_2, metadata_2 = await interaction.generate_response(
            instance_id, generated_messages_2
        )
        
        assert reward_2 == 1.0, f"Turn 2: Expected reward 1.0 for matching No, got {reward_2}"
        assert should_terminate_2 is True  # All turns completed
        assert metadata_2["current_turn_reward"] == 1.0
        assert metadata_2["turns_completed"] == 2
        assert metadata_2["average_reward"] == 1.0
        assert metadata_2["accuracy"] == 1.0
        assert metadata_2["num_matches"] == 2
    
    @pytest.mark.asyncio
    async def test_generate_response_yes_no_mismatch(self, interaction):
        """Test Yes/No answer mismatching across multiple turns."""
        # Create messages with Yes/No expected responses
        messages = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Is the sky blue?"},
            {"role": "assistant", "content": "Yes"},
            {"role": "user", "content": "Is the grass red?"},
            {"role": "assistant", "content": "No"},
        ]
        
        instance_id = await interaction.start_interaction(messages=messages)
        
        # Turn 1: Generated "No", expected "Yes" -> reward 0.0
        generated_messages_1 = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Is the sky blue?"},
            {"role": "assistant", "content": "No, it's not."},
        ]
        _, _, reward_1, metadata_1 = await interaction.generate_response(
            instance_id, generated_messages_1
        )
        
        assert reward_1 == 0.0, f"Turn 1: Expected reward 0.0 for mismatched Yes/No, got {reward_1}"
        assert metadata_1["average_reward"] == 0.0
        
        # Turn 2: Generated "Yes", expected "No" -> reward 0.0
        generated_messages_2 = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Is the sky blue?"},
            {"role": "assistant", "content": "No, it's not."},
            {"role": "user", "content": "Is the grass red?"},
            {"role": "assistant", "content": "Yes, definitely."},
        ]
        _, _, reward_2, metadata_2 = await interaction.generate_response(
            instance_id, generated_messages_2
        )
        
        assert reward_2 == 0.0, f"Turn 2: Expected reward 0.0 for mismatched Yes/No, got {reward_2}"
        assert metadata_2["turns_completed"] == 2
        assert metadata_2["average_reward"] == 0.0
        assert metadata_2["accuracy"] == 0.0
        assert metadata_2["num_matches"] == 0
    
    @pytest.mark.asyncio
    async def test_generate_response_yes_no_edge_cases(self, interaction):
        """Test Yes/No edge cases: both present, neither present, etc."""
        # Create messages with Yes/No expected responses
        messages = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Question 1?"},
            {"role": "assistant", "content": "Yes"},
            {"role": "user", "content": "Question 2?"},
            {"role": "assistant", "content": "No"},
            {"role": "user", "content": "Question 3?"},
            {"role": "assistant", "content": "Yes"},
        ]
        
        instance_id = await interaction.start_interaction(messages=messages)
        
        # Turn 1: Response contains both Yes and No -> reward 0.0
        generated_messages_1 = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Question 1?"},
            {"role": "assistant", "content": "Yes, but also no."},
        ]
        _, _, reward_1, metadata_1 = await interaction.generate_response(
            instance_id, generated_messages_1
        )
        
        assert reward_1 == 0.0, f"Turn 1: Expected reward 0.0 for both Yes and No present, got {reward_1}"
        
        # Turn 2: Response contains neither Yes nor No, but expected is No -> reward 0.0
        generated_messages_2 = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Question 1?"},
            {"role": "assistant", "content": "Yes, but also no."},
            {"role": "user", "content": "Question 2?"},
            {"role": "assistant", "content": "Maybe, I'm not sure."},
        ]
        _, _, reward_2, metadata_2 = await interaction.generate_response(
            instance_id, generated_messages_2
        )
        
        assert reward_2 == 0.0, f"Turn 2: Expected reward 0.0 for neither Yes nor No in response, got {reward_2}"
        
        # Turn 3: Correct matching with Yes -> reward 1.0
        generated_messages_3 = [
            {"role": "system", "content": "Answer with Yes or No."},
            {"role": "user", "content": "Question 1?"},
            {"role": "assistant", "content": "Yes, but also no."},
            {"role": "user", "content": "Question 2?"},
            {"role": "assistant", "content": "Maybe, I'm not sure."},
            {"role": "user", "content": "Question 3?"},
            {"role": "assistant", "content": "Absolutely yes!"},
        ]
        _, _, reward_3, metadata_3 = await interaction.generate_response(
            instance_id, generated_messages_3
        )
        
        assert reward_3 == 1.0, f"Turn 3: Expected reward 1.0 for matching Yes, got {reward_3}"
        assert metadata_3["turns_completed"] == 3
        assert metadata_3["num_matches"] == 1
        assert metadata_3["average_reward"] == pytest.approx(1.0 / 3.0)

    @pytest.mark.asyncio
    async def test_generate_response_invalid_instance(self, interaction):
        """Test error handling for invalid instance ID."""
        with pytest.raises(ValueError):
            await interaction.generate_response("invalid_id", [])
    
    @pytest.mark.asyncio
    async def test_generate_response_beyond_all_turns(self, interaction, sample_messages):
        """Test handling response after all turns completed."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        
        # Complete all turns
        for i in range(2):
            generated = sample_messages[: -1 - (1 - i) * 2] + [
                {"role": "assistant", "content": sample_messages[2 + i * 2]["content"]}
            ]
            await interaction.generate_response(instance_id, generated)
        
        # Try to generate response for non-existent turn
        should_terminate, next_prompt, reward, metadata = await interaction.generate_response(
            instance_id, sample_messages
        )
        
        assert should_terminate is True
        assert reward == 0.0
        assert "error" in metadata
    
    # ==================== Test calculate_score ====================
    
    @pytest.mark.asyncio
    async def test_calculate_score_perfect_match(self, interaction, sample_messages):
        """Test score calculation with perfect matches."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        
        # Complete interactions with exact matches
        for i in range(2):
            generated = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ]
            if i == 1:
                generated.extend([
                    {"role": "user", "content": "What is 3+3?"},
                    {"role": "assistant", "content": "3+3 equals 6."},
                ])
            await interaction.generate_response(instance_id, generated)
        
        score = await interaction.calculate_score(instance_id)
        assert score == 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_score_no_turns_completed(self, interaction, sample_messages):
        """Test score calculation with no turns completed."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        
        score = await interaction.calculate_score(instance_id)
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_score_invalid_instance(self, interaction):
        """Test error handling for invalid instance in calculate_score."""
        with pytest.raises(ValueError):
            await interaction.calculate_score("invalid_id")
    
    # ==================== Test finalize_interaction ====================
    
    @pytest.mark.asyncio
    async def test_finalize_interaction(self, interaction, sample_messages):
        """Test finalization of interaction."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        assert instance_id in interaction._instance_dict
        
        await interaction.finalize_interaction(instance_id)
        assert instance_id not in interaction._instance_dict
    
    @pytest.mark.asyncio
    async def test_finalize_nonexistent_interaction(self, interaction):
        """Test finalization of non-existent interaction (should not raise)."""
        # Should not raise an exception
        await interaction.finalize_interaction("nonexistent_id")
    
    # ==================== Test metadata structure ====================
    
    @pytest.mark.asyncio
    async def test_metadata_structure(self, interaction, sample_messages):
        """Test metadata structure and completeness."""
        instance_id = await interaction.start_interaction(messages=sample_messages)
        
        generated = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]
        
        _, _, _, metadata = await interaction.generate_response(instance_id, generated)
        
        # Check all expected keys are present
        expected_keys = {
            "turn_index", "turns_completed", "total_turns",
            "current_turn_reward", "all_turn_rewards", "num_matches",
            "average_reward", "accuracy", "should_terminate", "turn_history"
        }
        assert set(metadata.keys()) >= expected_keys
        
        # Check types
        assert isinstance(metadata["turn_index"], int)
        assert isinstance(metadata["turns_completed"], int)
        assert isinstance(metadata["total_turns"], int)
        assert isinstance(metadata["current_turn_reward"], float)
        assert isinstance(metadata["all_turn_rewards"], list)
        assert isinstance(metadata["average_reward"], float)
        assert isinstance(metadata["accuracy"], float)
        assert isinstance(metadata["should_terminate"], bool)
        assert isinstance(metadata["turn_history"], list)


if __name__ == "__main__":
    # Run tests with: pytest tests/interactions/test_multiturn_dialog_interaction.py
    pytest.main([__file__, "-v"])

