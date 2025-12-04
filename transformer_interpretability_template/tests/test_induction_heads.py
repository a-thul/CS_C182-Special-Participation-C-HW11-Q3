"""
Unit tests for induction head implementation.

This module tests the induction copy head mechanism including:
    - Basic pattern copying behavior
    - Correct token prediction for known sequences

References:
    - pytest documentation: https://docs.pytest.org/
    - Google Python Style Guide (testing): https://google.github.io/styleguide/pyguide.html#310-test-names
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from transformer_interpretability import induction_copy_head
from transformer_interpretability.utils.constants import EPSILON, VOCAB_SIZE


class TestInductionCopyHead:
    """Test suite for induction_copy_head function."""

    @pytest.fixture
    def sample_embeddings(self) -> list[list[float]]:
        """
        Create sample embeddings for sequence [a, b, c, d, a].

        Returns
        -------
        list[list[float]]
            2-hot encoded embeddings.
        """
        return [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # a at pos 0
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # b at pos 1
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # c at pos 2
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # d at pos 3
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # a at pos 4
        ]

    def test_basic_induction_pattern(
        self,
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test that induction head correctly predicts 'b' after seeing 'ab...a'.

        For sequence [a, b, c, d, a], the model should predict 'b' at the end
        because 'a' was previously followed by 'b'.
        """
        # Arrange
        attention_strength = 10.0
        expected_output = [0.000045, 0.999864, 0.000045, 0.000045]

        # Act
        output = induction_copy_head(sample_embeddings, attention_strength)

        # Assert
        np.testing.assert_allclose(
            output,
            expected_output,
            atol=EPSILON,
            err_msg="Induction head should predict 'b' after 'ab...a'",
        )

    def test_output_shape(self, sample_embeddings: list[list[float]]) -> None:
        """Test that output has correct shape (vocab_size,)."""
        # Act
        output = induction_copy_head(sample_embeddings, attention_strength=10.0)

        # Assert
        assert output.shape == (VOCAB_SIZE,), (
            f"Expected shape ({VOCAB_SIZE},), got {output.shape}"
        )

    def test_output_sums_to_one(self, sample_embeddings: list[list[float]]) -> None:
        """Test that output logits after softmax sum to 1.

        Note: The output is already in softmax form due to attention mechanism.
        """
        # Act
        output = induction_copy_head(sample_embeddings, attention_strength=10.0)

        # Assert
        assert np.isclose(np.sum(output), 1.0, atol=1e-3), (
            f"Output should sum to ~1.0, got {np.sum(output)}"
        )

    def test_higher_attention_strength_sharper_distribution(
        self,
        sample_embeddings: list[list[float]],
    ) -> None:
        """Test that higher attention strength creates sharper predictions."""
        # Act
        output_low = induction_copy_head(sample_embeddings, attention_strength=1.0)
        output_high = induction_copy_head(sample_embeddings, attention_strength=20.0)

        # Assert: higher strength should have higher max probability
        assert np.max(output_high) > np.max(output_low), (
            "Higher attention strength should create sharper distribution"
        )

    def test_returns_numpy_array(self, sample_embeddings: list[list[float]]) -> None:
        """Test that function returns numpy array."""
        # Act
        output = induction_copy_head(sample_embeddings, attention_strength=10.0)

        # Assert
        assert isinstance(output, np.ndarray), (
            f"Expected numpy.ndarray, got {type(output)}"
        )


def load_test_cases(filename: str) -> dict[str, Any]:
    """
    Load test cases from JSON file.

    Parameters
    ----------
    filename : str
        Name of the JSON file containing test cases.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping test case IDs to test data.
    """
    possible_paths = [
        Path(__file__).parent / "test_cases" / filename,
        Path(filename),
        Path("cs182fa25_public/hw11/code/q_coding_interpretability") / filename,
        Path.cwd() / filename,
    ]

    for path in possible_paths:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)

    pytest.skip(f"Test case file {filename} not found")
    return {}


@pytest.mark.parametrize("test_case_id", [f"Test case {i}" for i in range(1, 17)])
def test_induction_head_from_json(test_case_id: str) -> None:
    """
    Test induction_copy_head against provided test cases.

    Parameters
    ----------
    test_case_id : str
        Identifier for the test case.
    """
    test_cases = load_test_cases("induction_head_test_cases.json")

    if test_case_id not in test_cases:
        pytest.skip(f"Test case {test_case_id} not found in file")

    embeddings, attention_strength, expected_output = test_cases[test_case_id]
    output = induction_copy_head(embeddings, attention_strength).tolist()

    np.testing.assert_allclose(
        output,
        expected_output,
        atol=EPSILON,
        err_msg=f"{test_case_id} failed",
    )
