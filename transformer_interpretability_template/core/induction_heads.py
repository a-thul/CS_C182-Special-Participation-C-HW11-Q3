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

    The QK matrix should be designed so that each position attends strongly
    to its immediate predecessor. Think about which dimensions encode position
    and how to create a "shift-by-one" attention pattern.

    The OV matrix should copy the token identity from the attended position.
    Important: Write the output to dims 4-7 to avoid overwriting the original
    token embedding when adding the residual connection later.
    """
    wqk = np.zeros((d_model, d_model), dtype=np.float64)
    wov = np.zeros((d_model, d_model), dtype=np.float64)

    # WQK matrix: position i attends to position i-1
    # TODO: Set up the QK matrix so each position attends to its predecessor

    # OV matrix: copy token identity to a non-overlapping output location
    # TODO: Set up the OV matrix to copy token info to dims 4-7

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
    After the previous token head and residual connection, each position
    has information in two places:
        - Current token identity (from residual connection)
        - Previous token identity (from attention output)

    The QK matrix should match positions that have the same current token
    as another position's previous token. Think about which dimensions
    hold each piece of information.

    The OV matrix should copy the current token identity to produce logits.
    """
    wqk = np.zeros((d_model, d_model), dtype=np.float64)
    wov = np.zeros((d_model, d_model), dtype=np.float64)

    # QK matrix: match current token with previous token info
    # TODO: Set up matching between current token dims and previous token dims

    # OV matrix: copy token identity to produce output logits
    # TODO: Set up identity mapping for token dimensions

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

    1. **Previous Token Head**: Run attention that copies information about
       what token appeared at the previous position. After this step, each
       position knows about its predecessor's token identity.

    2. **Residual Connection**: Combine the attention output with the original
       token embeddings (but not position embeddings). This gives each position
       information about both its own token and the previous token.

    3. **Drop Position 0**: The first position has no valid predecessor, so
       exclude it from further processing.

    4. **Copying Head**: Run attention that looks for positions where the
       previous token matches the current query token. This finds the pattern
       "ab...a" and copies what followed "a" previously (which is "b").

    5. **Extract Logits**: Return the token logits from the final position.

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
    # Step 0: Convert to numpy array
    # TODO: Convert embeddings to numpy array and extract dimensions

    # Step 1: Build and run previous token head
    # TODO: Create previous token matrices and run attention

    # Step 2: Add residual connection (token info only, not position)
    # TODO: Combine attention output with original token embeddings

    # Step 3: Drop position 0 (no valid previous token)
    # TODO: Remove the first position from the combined representation

    # Step 4: Build and run copying head
    # TODO: Create copying matrices and run attention

    # Step 5: Return logits for the final position
    # TODO: Extract and return the final position's token logits

    raise NotImplementedError("Implement induction_copy_head")
