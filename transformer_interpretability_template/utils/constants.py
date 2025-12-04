"""
Constants and configuration values for transformer interpretability.

This module centralizes all magic numbers and configuration values
used throughout the package, following the principle of avoiding
scattered literals (Google Style Guide Section 2.14).

References:
    - Google Python Style Guide Section 2.14: Lexical Scoping
      https://google.github.io/styleguide/pyguide.html
"""

from typing import Final

# Numerical constants
EPSILON: Final[float] = 1e-6
"""Small value for numerical comparisons and stability."""

MASK_VALUE: Final[float] = -1e9
"""Large negative value for attention masking (approximates -infinity)."""

# Random seed for reproducibility
DEFAULT_SEED: Final[int] = 2025
"""Default random seed for reproducible experiments."""

# Vocabulary configuration for induction head examples
VOCAB_SIZE: Final[int] = 4
"""Number of tokens in the vocabulary (a, b, c, d)."""

MAX_SEQUENCE_LENGTH: Final[int] = 5
"""Maximum sequence length supported."""

D_MODEL: Final[int] = 9
"""Model dimension: vocab_size (4) + max_seq_len (5)."""

# Token mappings
TOKEN_TO_ID: Final[dict[str, int]] = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
}
"""Mapping from token strings to their numeric IDs."""

ID_TO_TOKEN: Final[dict[int, str]] = {v: k for k, v in TOKEN_TO_ID.items()}
"""Mapping from numeric IDs to token strings."""
