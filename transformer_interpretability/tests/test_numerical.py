"""
Unit tests for numerical utility functions.

This module tests the numerical utilities including:
    - Softmax numerical stability
    - Causal mask generation

References:
    - pytest documentation: https://docs.pytest.org/
"""

import numpy as np
import pytest

from transformer_interpretability.utils.numerical import create_causal_mask, softmax


class TestSoftmax:
    """Test suite for softmax function."""

    def test_basic_softmax(self) -> None:
        """Test softmax on simple input."""
        # Arrange
        logits = np.array([1.0, 2.0, 3.0])

        # Act
        probs = softmax(logits)

        # Assert
        assert np.isclose(np.sum(probs), 1.0), "Softmax should sum to 1"
        assert np.all(probs > 0), "All probabilities should be positive"

    def test_softmax_sums_to_one(self) -> None:
        """Test that softmax output sums to 1 along specified axis."""
        # Arrange
        logits = np.random.randn(3, 4)

        # Act
        probs = softmax(logits, axis=-1)

        # Assert
        np.testing.assert_allclose(
            np.sum(probs, axis=-1),
            np.ones(3),
            atol=1e-10,
            err_msg="Each row should sum to 1",
        )

    def test_numerical_stability_large_values(self) -> None:
        """Test softmax stability with large input values."""
        # Arrange: values that would overflow without max-subtraction
        logits = np.array([1000.0, 1001.0, 1002.0])

        # Act
        probs = softmax(logits)

        # Assert
        assert not np.any(np.isnan(probs)), "Should not produce NaN"
        assert not np.any(np.isinf(probs)), "Should not produce Inf"
        assert np.isclose(np.sum(probs), 1.0), "Should still sum to 1"

    def test_numerical_stability_negative_values(self) -> None:
        """Test softmax with large negative values."""
        # Arrange
        logits = np.array([-1000.0, -999.0, -998.0])

        # Act
        probs = softmax(logits)

        # Assert
        assert not np.any(np.isnan(probs)), "Should not produce NaN"
        assert np.isclose(np.sum(probs), 1.0), "Should still sum to 1"

    def test_softmax_2d_axis(self) -> None:
        """Test softmax on 2D array with different axes."""
        # Arrange
        logits = np.array([[1, 2], [3, 4]])

        # Act
        probs_row = softmax(logits, axis=-1)
        probs_col = softmax(logits, axis=0)

        # Assert
        np.testing.assert_allclose(
            np.sum(probs_row, axis=-1),
            [1.0, 1.0],
            err_msg="Row softmax should sum to 1 per row",
        )
        np.testing.assert_allclose(
            np.sum(probs_col, axis=0),
            [1.0, 1.0],
            err_msg="Column softmax should sum to 1 per column",
        )


class TestCreateCausalMask:
    """Test suite for create_causal_mask function."""

    def test_basic_mask_shape(self) -> None:
        """Test that mask has correct shape."""
        # Arrange
        seq_len = 5

        # Act
        mask = create_causal_mask(seq_len)

        # Assert
        assert mask.shape == (seq_len, seq_len)

    def test_diagonal_is_zero(self) -> None:
        """Test that diagonal elements are zero (can attend to self)."""
        # Arrange & Act
        mask = create_causal_mask(4)

        # Assert
        np.testing.assert_array_equal(
            np.diag(mask),
            np.zeros(4),
            err_msg="Diagonal should be zero",
        )

    def test_lower_triangular_is_zero(self) -> None:
        """Test that lower triangular part is zero (can attend to past)."""
        # Arrange & Act
        mask = create_causal_mask(4)

        # Assert
        lower = np.tril(mask)
        np.testing.assert_array_equal(
            lower,
            np.zeros((4, 4)),
            err_msg="Lower triangular should be zero",
        )

    def test_upper_triangular_is_masked(self) -> None:
        """Test that upper triangular part has mask value."""
        # Arrange
        mask_value = -1e9

        # Act
        mask = create_causal_mask(4, mask_value=mask_value)

        # Assert
        upper_indices = np.triu_indices(4, k=1)
        assert np.all(mask[upper_indices] == mask_value), (
            "Upper triangular should have mask value"
        )

    def test_custom_mask_value(self) -> None:
        """Test that custom mask value is applied."""
        # Arrange
        custom_value = -1e6

        # Act
        mask = create_causal_mask(3, mask_value=custom_value)

        # Assert
        assert mask[0, 1] == custom_value
        assert mask[0, 2] == custom_value
        assert mask[1, 2] == custom_value

    def test_invalid_seq_len_raises(self) -> None:
        """Test that non-positive seq_len raises ValueError."""
        with pytest.raises(ValueError, match="seq_len must be positive"):
            create_causal_mask(0)

        with pytest.raises(ValueError, match="seq_len must be positive"):
            create_causal_mask(-1)

    def test_size_one(self) -> None:
        """Test mask for single-element sequence."""
        # Act
        mask = create_causal_mask(1)

        # Assert
        np.testing.assert_array_equal(mask, [[0.0]])
