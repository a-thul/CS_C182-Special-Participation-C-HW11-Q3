"""
Single attention head implementation.

This module provides a clean, well-documented implementation of a single
transformer attention head with causal masking for autoregressive models.

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

    1. **Pre-attention scores**: Compute how much each position should attend
       to every other position by combining the input with the query-key matrix.
       The result is a (seq_len, seq_len) matrix where entry (i, j) measures
       how much position i attends to position j.

    2. **Causal masking**: Prevent positions from attending to future positions
       by masking out the upper triangular portion of the attention matrix.
       This enforces the autoregressive property.

    3. **Attention weights**: Convert the masked scores into a probability
       distribution using softmax. Each row should sum to 1.

    4. **Value projection**: Project the input embeddings through the
       output-value matrix to create value vectors.

    5. **Output**: Compute the final output as a weighted combination of
       value vectors, where the weights come from the attention distribution.

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
    # Step 0: Convert inputs to numpy arrays
    # TODO: Convert attn_input, wqk, wov to numpy arrays with dtype=np.float64
    # TODO: Extract seq_len and d_model from the input shape

    # Step 1: Compute pre-attention scores
    # TODO: Compute the attention score matrix

    # Step 2: Apply causal mask (prevent attending to future positions)
    # TODO: Create and apply causal mask to the scores

    # Step 3: Apply softmax to get attention weights (row-wise)
    # TODO: Apply softmax to get probability distributions

    # Step 4: Project inputs through output-value matrix
    # TODO: Compute value projections

    # Step 5: Compute attention-weighted output
    # TODO: Compute the final output

    raise NotImplementedError("Implement single_attention_head")
