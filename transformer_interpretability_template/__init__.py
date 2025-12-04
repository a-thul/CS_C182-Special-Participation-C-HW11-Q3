"""
Transformer Interpretability Package.

A modular implementation of transformer attention mechanisms for educational purposes,
demonstrating single attention heads and induction copy heads.

This package provides implementations for:
    - Single attention head computation with causal masking
    - Induction copy head mechanism combining previous token and copying heads

Example:
    >>> from transformer_interpretability import single_attention_head, induction_copy_head
    >>> import numpy as np
    >>> # Create simple input
    >>> attn_input = [[0, 1], [1, 1], [1, 2]]
    >>> WQK = [[1, 1], [0, 0]]
    >>> WOV = [[1, 1], [0, 0]]
    >>> output = single_attention_head(attn_input, WQK, WOV)

References:
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 484: https://peps.python.org/pep-0484/
    - Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
    - NumPy Style Guide: https://numpydoc.readthedocs.io/en/latest/format.html
"""

from transformer_interpretability.core.attention import single_attention_head
from transformer_interpretability.core.induction_heads import induction_copy_head

__all__ = ["single_attention_head", "induction_copy_head"]
__version__ = "1.0.0"
