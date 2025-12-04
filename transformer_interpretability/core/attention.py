"""
Single attention head implementation.

This module provides a clean, well-documented implementation of a single
transformer attention head with causal masking for autoregressive models.

The implementation follows the standard attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

With the simplification that Q, K, V projections are pre-multiplied:
    - WQK = W_Q^T @ W_K (query-key projection)
    - WOV = W_O^T @ W_V (output-value projection)

References:
    - Vaswani et al., "Attention Is All You Need" (2017)
      https://arxiv.org/abs/1706.03762
    - Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)
      https://transformer-circuits.pub/2021/framework/index.html
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from transformer_interpretability.utils.numerical import create_causal_mask, softmax


# Type alias for input flexibility
MatrixInput = Union[list[list[float]], NDArray[np.floating]]


def single_attention_head(
    attn_input: MatrixInput,
    wqk: MatrixInput,
    wov: MatrixInput,
) -> NDArray[np.floating]:
    """
    Compute the output of a single causal attention head.

    Implements a single attention head with pre-multiplied projection matrices
    and causal masking. This is a simplified but mathematically equivalent
    formulation commonly used in mechanistic interpretability research.

    Parameters
    ----------
    attn_input : MatrixInput
        Input embeddings of shape (seq_len, d_model).
        Each row is a token embedding vector.
    wqk : MatrixInput
        Pre-multiplied query-key matrix of shape (d_model, d_model).
        Represents W_Q^T @ W_K in standard attention notation.
    wov : MatrixInput
        Pre-multiplied output-value matrix of shape (d_model, d_model).
        Represents W_O^T @ W_V in standard attention notation.

    Returns
    -------
    NDArray[np.floating]
        Attention head output of shape (seq_len, d_model).
        Each row contains the attention-weighted combination of value vectors.

    Notes
    -----
    The computation proceeds as follows:

    1. **Pre-attention scores**: Z = X @ WQK @ X^T
       - Shape: (seq_len, seq_len)
       - Entry (i, j) measures how much position i attends to position j

    2. **Causal masking**: Set Z[i, j] = -inf for j > i
       - Prevents attending to future positions

    3. **Attention weights**: A = softmax(Z, axis=-1)
       - Row-wise softmax to get probability distribution

    4. **Value projection**: V = X @ WOV
       - Project inputs through output-value matrix

    5. **Output**: O = A @ V
       - Weighted sum of value vectors

    Examples
    --------
    >>> import numpy as np
    >>> attn_input = [[0, 1], [1, 1], [1, 2]]
    >>> WQK = [[1, 1], [0, 0]]
    >>> WOV = [[1, 1], [0, 0]]
    >>> output = single_attention_head(attn_input, WQK, WOV)
    >>> output.shape
    (3, 2)

    References
    ----------
    .. [1] Vaswani, A., et al. "Attention is all you need."
           Advances in neural information processing systems 30 (2017).
    """
    # Convert inputs to numpy arrays (defensive copy for immutability)
    x = np.asarray(attn_input, dtype=np.float64)
    wqk_arr = np.asarray(wqk, dtype=np.float64)
    wov_arr = np.asarray(wov, dtype=np.float64)

    seq_len, d_model = x.shape

    # Step 1: Compute pre-attention scores
    # Z[i, j] = (x[i] @ WQK) @ x[j]^T = query_i dot key_j
    pre_attn_scores = x @ wqk_arr @ x.T

    # Step 2: Apply causal mask (prevent attending to future positions)
    causal_mask = create_causal_mask(seq_len)
    masked_scores = pre_attn_scores + causal_mask

    # Step 3: Apply softmax to get attention weights (row-wise)
    attn_weights = softmax(masked_scores, axis=-1)

    # Step 4: Project inputs through output-value matrix
    value_projections = x @ wov_arr.T

    # Step 5: Compute attention-weighted output
    output = attn_weights @ value_projections

    return output
