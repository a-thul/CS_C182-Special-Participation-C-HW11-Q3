"""
Core transformer interpretability modules.

This subpackage contains the main implementations for attention mechanisms
and induction head patterns used in transformer interpretability studies.
"""

from transformer_interpretability.core.attention import single_attention_head
from transformer_interpretability.core.induction_heads import induction_copy_head

__all__ = ["single_attention_head", "induction_copy_head"]
