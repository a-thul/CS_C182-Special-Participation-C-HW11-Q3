"""
Induction head implementation.

This module implements the induction head mechanism, a key pattern discovered
in transformer interpretability research. Induction heads enable in-context
learning by identifying and copying patterns from earlier in the sequence.

The mechanism predicts that if pattern "ab" appeared earlier, then seeing "a"
again suggests "b" will follow. This is implemented through two heads:

1. **Previous Token Head**: Copies information about what token preceded each position
2. **Copying Head**: Matches current context with historical patterns and copies

References:
    - Olsson et al., "In-context Learning and Induction Heads" (2022)
      https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
    - Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)
      https://transformer-circuits.pub/2021/framework/index.html
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from transformer_interpretability.core.attention import single_attention_head


# Type alias for input flexibility
MatrixInput = Union[list[list[float]], NDArray[np.floating]]

# Module-level constants
_VOCAB_SIZE = 4
_MAX_SEQ_LEN = 5
_D_MODEL = 9  # 4 (vocab) + 5 (position)


def _build_previous_token_matrices(
    d_model: int,
    attention_strength: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Construct weight matrices for the previous token head.

    The previous token head attends from each position to its immediate
    predecessor using positional information, then extracts token identity.

    Parameters
    ----------
    d_model : int
        Model dimension (token dims + position dims).
    attention_strength : float
        Scaling factor for attention scores in QK matrix.

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        WQK and WOV matrices, each of shape (d_model, d_model).

    Notes
    -----
    For a 2-hot encoding with vocab_size=4 and max_seq_len=5:
        - Dims 0-3: token identity (one-hot)
        - Dims 4-8: position (one-hot)

    The QK matrix is designed so position i attends to position i-1:
        - Query uses position p, Key uses position p-1
        - This creates a "shift-by-one" attention pattern

    The OV matrix copies token identity (dims 0-3) to output.
    """
    wqk = np.zeros((d_model, d_model), dtype=np.float64)
    wov = np.zeros((d_model, d_model), dtype=np.float64)

    # Position dims: 4, 5, 6, 7, 8 for positions 0, 1, 2, 3, 4
    # For position i to attend to position i-1:
    # Query at position i (dim 4+i) should match Key at position i-1 (dim 4+i-1)
    # So WQK[4+i, 4+i-1] = attention_strength for i in 1..4
    for pos in range(1, _MAX_SEQ_LEN):
        query_dim = _VOCAB_SIZE + pos      # Current position dimension
        key_dim = _VOCAB_SIZE + pos - 1    # Previous position dimension
        wqk[query_dim, key_dim] = attention_strength

    # OV matrix: copy token identity (dims 0-3) to dims 4-7 in output
    for token_dim in range(_VOCAB_SIZE):
        wov[_VOCAB_SIZE + token_dim, token_dim] = 1.0

    return wqk, wov


def _build_copying_matrices(
    d_model: int,
    vocab_size: int,
    attention_strength: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Construct weight matrices for the copying head.

    The copying head matches tokens based on (current_token, previous_token)
    pairs and copies the token that followed matching historical patterns.

    Parameters
    ----------
    d_model : int
        Model dimension.
    vocab_size : int
        Number of tokens in vocabulary.
    attention_strength : float
        Scaling factor for attention scores in QK matrix.

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        WQK and WOV matrices, each of shape (d_model, d_model).

    Notes
    -----
    After the previous token head, each position has information about:
        - Its own token identity (from residual connection)
        - The previous token's identity (from attention output)

    The copying head's QK matrix matches positions with the same
    (current, previous) token pair. The OV matrix copies the current
    token identity to produce logits.
    """
    wqk = np.zeros((d_model, d_model), dtype=np.float64)
    wov = np.zeros((d_model, d_model), dtype=np.float64)

    # QK matrix: match current token (query dims 0-3) with previous token (key dims 4-7)
    for token_dim in range(vocab_size):
        wqk[token_dim, vocab_size + token_dim] = attention_strength

    # OV matrix: copy token identity to produce output logits
    for token_dim in range(vocab_size):
        wov[token_dim, token_dim] = 1.0

    return wqk, wov


def induction_copy_head(
    embeddings: MatrixInput,
    attention_strength: float,
) -> NDArray[np.floating]:
    """
    Compute induction head output for next token prediction.

    Implements the full induction head mechanism by combining a previous
    token head with a copying head. This enables the model to predict
    that patterns like "ab...a" will be followed by "b".

    Parameters
    ----------
    embeddings : MatrixInput
        2-hot encoded input embeddings of shape (seq_len, d_model).
        Format: [token_onehot (4 dims) | position_onehot (5 dims)]
        Vocabulary: a=0, b=1, c=2, d=3
        Positions: 0-4
    attention_strength : float
        Scaling factor for attention. Higher values create sharper
        attention distributions.

    Returns
    -------
    NDArray[np.floating]
        Logits for next token prediction, shape (vocab_size,) = (4,).
        Indices correspond to tokens [a, b, c, d].

    Notes
    -----
    The induction mechanism works as follows:

    1. **Previous Token Head**: For each position i, attends to position i-1
       and copies its token identity to the output.

    2. **Residual Connection**: Add token identity (not position) from the
       original embedding to previous token head output.

    3. **Drop Position 0**: Position 0 has no valid predecessor, so we
       exclude it from further processing.

    4. **Copying Head**: At the final position, looks for earlier positions
       with matching (current_token, previous_token) patterns and copies
       the token that followed those patterns.

    Examples
    --------
    >>> # Sequence: [a, b, c, d, a]
    >>> # Pattern: 'a' at position 0 is followed by 'b'
    >>> # At position 4, seeing 'a' should predict 'b'
    >>> embeddings = [
    ...     [1, 0, 0, 0, 1, 0, 0, 0, 0],  # a at pos 0
    ...     [0, 1, 0, 0, 0, 1, 0, 0, 0],  # b at pos 1
    ...     [0, 0, 1, 0, 0, 0, 1, 0, 0],  # c at pos 2
    ...     [0, 0, 0, 1, 0, 0, 0, 1, 0],  # d at pos 3
    ...     [1, 0, 0, 0, 0, 0, 0, 0, 1],  # a at pos 4
    ... ]
    >>> logits = induction_copy_head(embeddings, attention_strength=10.0)
    >>> predicted_token = np.argmax(logits)  # Should be 1 (token 'b')

    References
    ----------
    .. [1] Olsson, C., et al. "In-context learning and induction heads."
           arXiv preprint arXiv:2209.11895 (2022).
    """
    # Convert to numpy array
    x = np.asarray(embeddings, dtype=np.float64)
    seq_len, d_model = x.shape

    # Build and run previous token head
    wqk_prev, wov_prev = _build_previous_token_matrices(d_model, attention_strength)
    prev_token_output = single_attention_head(x, wqk_prev, wov_prev)

    # Add residual connection (token info only, not position)
    # This gives each position info about: (its token, previous token)
    combined = prev_token_output.copy()
    combined[:, :_VOCAB_SIZE] += x[:, :_VOCAB_SIZE]

    # Drop position 0 (no valid previous token)
    combined = combined[1:, :]

    # Build and run copying head
    wqk_copy, wov_copy = _build_copying_matrices(d_model, _VOCAB_SIZE, attention_strength)
    copy_output = single_attention_head(combined, wqk_copy, wov_copy)

    # Return logits for the final position
    final_logits = copy_output[-1, :_VOCAB_SIZE]

    return final_logits
