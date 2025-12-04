"""
Unit tests for single attention head implementation.

This module tests the core attention mechanism including:
    - Basic attention computation
    - Causal masking behavior
    - Numerical correctness

References:
    - pytest documentation: https://docs.pytest.org/
    - Google Python Style Guide (testing): https://google.github.io/styleguide/pyguide.html#310-test-names
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from transformer_interpretability import single_attention_head
from transformer_interpretability.utils.constants import EPSILON


class TestSingleAttentionHead:
    """Test suite for single_attention_head function."""

    def test_basic_attention_output(self) -> None:
        """Test basic attention computation with simple inputs.

        Verifies that the attention head produces correct outputs
        for a minimal example with known expected results.
        """
        # Arrange
        attn_input = [[0, 1], [1, 1], [1, 2]]
        wqk = [[1, 1], [0, 0]]
        wov = [[1, 1], [0, 0]]
        expected_output = [
            [1.0, 0.0],
            [1.7310585786300048, 0.0],
            [2.5752103826044417, 0.0],
        ]

        # Act
        output = single_attention_head(attn_input, wqk, wov)

        # Assert
        np.testing.assert_allclose(
            output,
            expected_output,
            atol=EPSILON,
            err_msg="Basic attention output mismatch",
        )

    def test_output_shape(self) -> None:
        """Test that output shape matches input sequence length and model dimension."""
        # Arrange
        seq_len, d_model = 5, 8
        attn_input = np.random.randn(seq_len, d_model)
        wqk = np.random.randn(d_model, d_model)
        wov = np.random.randn(d_model, d_model)

        # Act
        output = single_attention_head(attn_input, wqk, wov)

        # Assert
        assert output.shape == (seq_len, d_model), (
            f"Expected shape ({seq_len}, {d_model}), got {output.shape}"
        )

    def test_causal_masking(self) -> None:
        """Test that causal masking prevents attending to future positions.

        Verifies that the first position's output only depends on itself,
        not on any future positions.
        """
        # Arrange: identity matrices so attention directly reveals weights
        d_model = 3
        attn_input = np.eye(d_model)  # Each position has unique embedding
        wqk = np.eye(d_model)
        wov = np.eye(d_model)

        # Act
        output = single_attention_head(attn_input, wqk, wov)

        # Assert: first position should only attend to itself
        # With identity WOV, output[0] should equal input[0]
        np.testing.assert_allclose(
            output[0],
            attn_input[0],
            atol=EPSILON,
            err_msg="First position should only attend to itself",
        )

    def test_accepts_list_input(self) -> None:
        """Test that function accepts Python lists as input."""
        # Arrange
        attn_input = [[1.0, 0.0], [0.0, 1.0]]
        wqk = [[1.0, 0.0], [0.0, 1.0]]
        wov = [[1.0, 0.0], [0.0, 1.0]]

        # Act & Assert (should not raise)
        output = single_attention_head(attn_input, wqk, wov)
        assert isinstance(output, np.ndarray)

    def test_returns_numpy_array(self) -> None:
        """Test that function returns numpy array."""
        # Arrange
        attn_input = [[1.0, 0.0], [0.0, 1.0]]
        wqk = [[1.0, 0.0], [0.0, 1.0]]
        wov = [[1.0, 0.0], [0.0, 1.0]]

        # Act
        output = single_attention_head(attn_input, wqk, wov)

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
    # Look for test cases in common locations
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


@pytest.mark.parametrize("test_case_id", [f"Test case {i}" for i in range(1, 50)])
def test_single_attention_head_from_json(test_case_id: str) -> None:
    """
    Test single_attention_head against provided test cases.

    Parameters
    ----------
    test_case_id : str
        Identifier for the test case.
    """
    test_cases = load_test_cases("single_attention_head_test_cases.json")

    if test_case_id not in test_cases:
        pytest.skip(f"Test case {test_case_id} not found in file")

    attn_input, wqk, wov, expected_output = test_cases[test_case_id]
    output = single_attention_head(attn_input, wqk, wov).tolist()

    np.testing.assert_allclose(
        output,
        expected_output,
        atol=EPSILON,
        err_msg=f"{test_case_id} failed",
    )
