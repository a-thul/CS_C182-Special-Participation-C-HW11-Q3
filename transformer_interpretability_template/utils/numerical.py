"""
Numerical utility functions for transformer computations.

This module provides numerically stable implementations of common operations
used in transformer attention mechanisms.

References:
    - Softmax numerical stability: https://cs231n.github.io/linear-classify/#softmax
    - NumPy Style Guide: https://numpydoc.readthedocs.io/en/latest/format.html
"""

import numpy as np
from numpy.typing import NDArray


def softmax(
    logits: NDArray[np.floating],
    axis: int = -1,
) -> NDArray[np.floating]:
    """
    Compute numerically stable softmax over the specified axis.

    Implements the softmax function with the max-subtraction trick to prevent
    numerical overflow when dealing with large logit values.

    Parameters
    ----------
    logits : NDArray[np.floating]
        Input array of logits. Can be any shape.
    axis : int, optional
        Axis along which to compute softmax, by default -1 (last axis).

    Returns
    -------
    NDArray[np.floating]
        Softmax probabilities with the same shape as input.
        Values along the specified axis sum to 1.

    Notes
    -----
    The implementation uses the identity:
        softmax(x) = softmax(x - max(x))

    This prevents overflow when exponentiating large values while
    maintaining numerical precision.

    Examples
    --------
    >>> import numpy as np
    >>> logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    >>> probs = softmax(logits, axis=-1)
    >>> np.allclose(probs.sum(axis=-1), 1.0)
    True

    References
    ----------
    .. [1] CS231n: Convolutional Neural Networks for Visual Recognition
           https://cs231n.github.io/linear-classify/#softmax
    """
    # Subtract max for numerical stability (PEP 8: clear variable naming)
    logits_shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def create_causal_mask(
    seq_len: int,
    mask_value: float = -1e9,
) -> NDArray[np.floating]:
    """
    Create a causal (lower-triangular) attention mask.

    Generates a mask that prevents attention to future positions,
    enforcing the autoregressive property in decoder-style attention.

    Parameters
    ----------
    seq_len : int
        Length of the sequence. Must be positive.
    mask_value : float, optional
        Value to use for masked (future) positions, by default -1e9.
        This large negative value ensures near-zero attention weights
        after softmax.

    Returns
    -------
    NDArray[np.floating]
        Square mask matrix of shape (seq_len, seq_len).
        Entry (i, j) is 0.0 if j <= i (can attend), else mask_value.

    Raises
    ------
    ValueError
        If seq_len is not positive.

    Examples
    --------
    >>> mask = create_causal_mask(3)
    >>> print(mask)
    [[ 0.e+00 -1.e+09 -1.e+09]
     [ 0.e+00  0.e+00 -1.e+09]
     [ 0.e+00  0.e+00  0.e+00]]

    Notes
    -----
    The mask is designed to be added to attention scores before softmax.
    Positions with mask_value will have near-zero attention weights.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    # Create upper triangular mask (True for positions to mask)
    # np.triu with k=1 gives strictly upper triangular (above diagonal)
    mask_indices = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    # Initialize with zeros and apply mask value
    mask = np.zeros((seq_len, seq_len), dtype=np.float64)
    mask[mask_indices] = mask_value

    return mask
